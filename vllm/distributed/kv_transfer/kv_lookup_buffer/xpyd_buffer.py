"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Non-blocking `insert`, blocking `drop_select`.
    - Use CPU signal pipe to avoid racing condition
    - Handles buffer size constraints and provide backpressure mechanism to 
      stop the prefill instance when the decode instance is slow.
"""
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Union
import queue
import torch
import zmq
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.distributed.kv_transfer.data import PullKey
from vllm.logger import init_logger

logger = init_logger(__name__)


class XpYdBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float):
        """
        signal_pipe: on CPU 
        
        NOTE: on-device recv will block all threads in the process, making the 
        KV cache producer unable to listen to new request while transmitting 
        KV cache. Luckily CPU recv only blocks the current thread so we use 
        CPU recv to listen to new request.
        
        data_pipe: on device (e.g. GPU)
        """

        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_lock = threading.Lock()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        
        self.kv_rank = self.data_pipe.kv_rank
        self.kv_role = self.data_pipe.config.kv_role
        self.zmq_ports = self.data_pipe.config.zmq_ports
        self.zmq_ip = self.data_pipe.config.zmq_ip
        self.producer_num = self.data_pipe.config.producer_num
        self.consumer_num = self.data_pipe.config.consumer_num
        self.context = zmq.Context()
        
        # self.d_rank_queue = queue.Queue() # 用于向request_handling_thread线程传递d_rank 
        if self.kv_role == "kv_producer":
            self.request_handling_threads: Optional[dict[threading.Thread]] = {}
            self.p_socket = self.context.socket(zmq.PULL)
            self.p_socket.bind(f"tcp://{self.zmq_ip}:{self.zmq_ports[self.kv_rank]}")
            
            self.queues = {
                self.hash_rank([self.kv_rank,x]):queue.Queue() 
                for x in range(self.producer_num, self.producer_num + self.consumer_num)
            }
            logger.info(f"[rank:{self.kv_rank}] self.queues={self.queues}")
            
            # 启动分发线程 和 处理线程
            self.request_distribute_thread: Optional[threading.Thread] = None
            self.request_distribute_thread = threading.Thread(target=self.drop_select_distribute)
            self.request_distribute_thread.start()
            
            self.ppd_pair_list = [[self.kv_rank,x] for x in range(self.producer_num, self.producer_num + self.consumer_num)]
            for pd_pair in self.ppd_pair_list:
                thread_idx = self.hash_rank(pd_pair=pd_pair)
                self.request_handling_threads[thread_idx] = threading.Thread(target=self.drop_select_handler,
                                                                             args=(pd_pair,))
                self.request_handling_threads[thread_idx].start()
            logger.info(f"[rank:{self.kv_rank}] self.request_handling_threads={self.request_handling_threads}")
            
        elif self.kv_role == "kv_consumer":
            # self.d_sockets = {}
            self.d_socket = None 
            self.dpd_pair_list = [[x,self.kv_rank] for x in range(0,self.producer_num)]
            for pd_pair in self.dpd_pair_list:
                p_rank = pd_pair[0]
                socket = self.context.socket(zmq.PUSH)
                logger.info(f"[rank:{self.kv_rank}][pd_pair:{pd_pair}] zmq_port={self.zmq_ports[p_rank]}")
                socket.connect(f"tcp://{self.zmq_ip}:{self.zmq_ports[p_rank]}")
                socket_idx = self.hash_rank(pd_pair=pd_pair)
                # self.d_sockets[socket_idx] = socket
                self.d_socket = socket
            # logger.info(f"[rank:{self.kv_rank}][self.d_sockets:{self.d_sockets}]")
            logger.info(f"[rank:{self.kv_rank}][self.d_socket:{self.d_socket}]")
            
            # self.d_socket = self.context.socket(zmq.PUSH)
            # logger.info(f"[rank:{self.kv_rank}] zmq_port={self.zmq_ports[0]}")
            # self.d_socket.connect(f"tcp://{self.zmq_ip}:{self.zmq_ports[0]}")

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None
        
    def hash_rank(self, pd_pair: List[int]):
        p_rank = pd_pair[0]
        d_rank = pd_pair[1]
        return p_rank * 100 + d_rank * 10

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)
        
        # recver表示D实例那边的tensor
        # sender表示P实例这边的tensor
        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0
    
    def _matches_no_roi(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)
        
        # recver表示D实例那边的tensor
        # sender表示P实例这边的tensor
        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        # roi_sender = tokens_roi_sender[1]
        # roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        # tokens_sender = tokens_sender[roi_sender]
        # tokens_recver = tokens_recver[roi_recver]
        

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length].to(self.data_pipe.device),
                          tokens_recver[:min_length].to(self.data_pipe.device)):
            return min_length

        return 0

    def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor], 
                                  d_rank: int) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        self.data_pipe.send_tensor(tensor, d_rank)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [input_tokens, roi, key, value, hidden]

        with self.buffer_lock:
            for data in buffer_item:
                self.buffer_size += self._get_element_size(data)
            self.buffer.append(buffer_item)

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self, pd_pair):
        queue_idx = self.hash_rank(pd_pair=pd_pair)
        d_rank = pd_pair[1]
        try:
            while True:
                input_tokens = self.queues[queue_idx].get()
                logger.info(f"[kv_rank:{self.kv_rank}][pd_pair:{pd_pair}]  recv input_tokens={input_tokens}")
                # logger.info(f"[kv_rank:{self.kv_rank}][pd_pair:{pd_pair}]  recv roi={roi}")
                
                # assert roi is not None, "Please provide the roi when sending "\
                #     "drop-select request"
                # roi = (roi > 0.5)
                # tokens_roi_recver = [input_tokens, roi]
                tokens_roi_recver = [input_tokens]
                matched_length = 0
                with self.buffer_lock:
                    for _ in range(len(self.buffer)):
                        # temp_length = self._matches(self.buffer[0],
                        #                             tokens_roi_recver)
                        temp_length = self._matches_no_roi(self.buffer[0],
                                                    tokens_roi_recver)
                        if temp_length > 0:
                            matched_length = temp_length
                            break
                        # rotate the element we just accessed to the end
                        self.buffer.rotate(-1)
                    if matched_length > 0:
                        # need to clone the tensor
                        # in case the tensor is freed before sending finishes
                        matched_item = self.buffer.popleft()
                        for tensor in matched_item:
                            self._send_tensor_and_dec_size(tensor, d_rank)
                    else:
                        # no match, just send None
                        for _ in range(5):
                            self.data_pipe.send_tensor(None, d_rank)
        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")
        
    def drop_select_distribute(self):
        try:
            while True:
                pull_key = self.p_socket.recv_pyobj()
                pd_pair = pull_key.pd_pair
                input_tokens = pull_key.input_tokens

                # 分发到不同的request_handle_thread处理
                request_handle_idx = self.hash_rank(pd_pair=pd_pair)
                self.queues[request_handle_idx].put(input_tokens)
        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e
        logger.debug("Closing drop_select_distribute")

    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor], pd_pair: int) -> List[Optional[torch.Tensor]]:

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone().float()

        p_rank = pd_pair[0] # sc_pd
        # self.request_queues[self.hash_rank(pd_pair)].put(pd_pair)
        pull_key = PullKey(pd_pair=pd_pair, input_tokens=input_tokens)
        # self.d_sockets[self.hash_rank(pd_pair)].send_pyobj(pull_key)
        self.d_socket.send_pyobj(pull_key)
        
        # self.signal_pipe.send_tensor(self.normal_signal,p_rank)
        # self.data_pipe.send_tensor(input_tokens,p_rank)
        # self.data_pipe.send_tensor(roi,p_rank)

        input_tokens = self.data_pipe.recv_tensor(p_rank)
        roi = self.data_pipe.recv_tensor(p_rank)
        if roi is not None:
            # convert from float tensor to bool tensor
            # as PyNccl does not support sending bool tensor
            roi = (roi > 0.5)
        key = self.data_pipe.recv_tensor(p_rank)
        value = self.data_pipe.recv_tensor(p_rank)
        hidden = self.data_pipe.recv_tensor(p_rank)

        return [input_tokens, roi, key, value, hidden]
    

    def full_handler(self):
        time.sleep(0.001)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor, pd_pair: int) -> None:

        if self.buffer_size > self.buffer_size_threshold:
            # log outside the while loop to avoid this message being logged
            # repeatedly.
            logger.debug("KV transfer buffer is full. Handling...")
        while self.buffer_size > self.buffer_size_threshold:
            self.full_handler()

        self._add_to_buffer(input_tokens, roi, key, value, hidden)
        
        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        # if self.request_handling_thread is None:
        #     self.request_handling_thread = threading.Thread(
        #         target=self.drop_select_handler)
        #     self.request_handling_thread.start()
        
        # sc_pd
        # self.d_rank_queue.put(d_rank)
        # if self.hash_rank(pd_pair) not in self.request_queues:    
        #     self.request_queues[self.hash_rank(pd_pair)] = queue.Queue()
        if self.hash_rank(pd_pair) not in self.request_handling_threads:
            thread = threading.Thread(target=self.drop_select_handler,args=(pd_pair,))
            thread.start()
            self.request_handling_threads[self.hash_rank(pd_pair)] = thread
        # self.request_queues[self.hash_rank(pd_pair)].put(pd_pair)
        
    def insert_zmq(self, input_tokens: torch.Tensor, roi: torch.Tensor,
            key: torch.Tensor, value: torch.Tensor,
            hidden: torch.Tensor, pd_pair: int) -> None:

        if self.buffer_size > self.buffer_size_threshold:
            # log outside the while loop to avoid this message being logged
            # repeatedly.
            logger.debug("KV transfer buffer is full. Handling...")
        while self.buffer_size > self.buffer_size_threshold:
            self.full_handler()

        self._add_to_buffer(input_tokens, roi, key, value, hidden)
        
        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        # if self.request_handling_thread is None:
        #     self.request_handling_thread = threading.Thread(
        #         target=self.drop_select_handler)
        #     self.request_handling_thread.start()
        
        # sc_pd
        # self.d_rank_queue.put(d_rank)
        # if self.hash_rank(pd_pair) not in self.request_queues:    
        #     self.request_queues[self.hash_rank(pd_pair)] = queue.Queue()
        if self.hash_rank(pd_pair) not in self.request_handling_threads:
            thread = threading.Thread(target=self.drop_select_handler,args=(pd_pair,))
            thread.start()
            self.request_handling_threads[self.hash_rank(pd_pair)] = thread
        # self.request_queues[self.hash_rank(pd_pair)].put(pd_pair)

    def close(self):

        if hasattr(self, "request_handling_threads"
                   ) and len(self.request_handling_threads) != 0:
            for _, value in self.request_handling_threads.items():
                value.join()
        # if hasattr(self, "request_handling_thread"
        #            ) and self.request_handling_thread:
        #     self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
