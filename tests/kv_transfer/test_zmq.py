import torch 
import zmq
import os, sys 
from typing import Deque, List, Optional
import threading 
from collections import deque
import time
import threading 

port_list = [12300,12301,12302]

class SimpleBuffer():
    def __init__(self, rank):
        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_lock = threading.Lock()
        self.kv_rank = rank
        self.context = zmq.Context()
        if rank == 0:
            # self.sockets = [] 
            # for i in range(3):
            self.socket = self.context.socket(zmq.PULL)
            print(f"[rank:{rank}]")
            # socket.bind(f"tcp://127.0.0.1:{port_list[i]}")
            self.socket.bind(f"tcp://127.0.0.1:12300")
                # self.sockets.append(socket)
        else:
            self.socket = self.context.socket(zmq.PUSH)
            print(f"[rank:{rank}]")
            # self.socket.connect(f"tcp://127.0.0.1:{port_list[rank-1]}")
            self.socket.connect(f"tcp://127.0.0.1:12300")
            
    def __exit__(self, exc_type, exc_value, traceback):
        if self.kv_rank == 0:
            for socket in self.sockets:
                socket.close()
        else:
            self.socket.close()
        self.context.term()

    
    def hash_rank(self, pd_pair: List[int]):
        p_rank = pd_pair[0]
        d_rank = pd_pair[1]
        return p_rank * 100 + d_rank * 10
    
    def recv_and_handle(self):
        print("接收线程启动成功")
        while True:
            # message = self.sockets[rank].recv_string()
            message = self.socket.recv_string()
            print(message)
    
def main():
    rank = int(sys.argv[1])
    # print(f"[rank:{rank}]")
    buffer = SimpleBuffer(rank)
    if rank == 0:
        # rank1_res = buffer.sockets[0].recv_string()
        # rank2_res = buffer.sockets[1].recv_string()
        # print(f"{rank1_res = }")
        # print(f"{rank2_res = }")
        
        # time.sleep(1)
        # rank3_res = buffer.sockets[2].recv_string()
        # rank3_res = buffer.sockets[2].recv_string()
        # print(f"{rank3_res = }")
        
        # threads = []
        # for i in range(len(buffer.sockets)):
        #     thread = threading.Thread(target=buffer.recv_and_handle, args=(i,))
        #     thread.start()
        #     threads.append(thread)
            
        # time.sleep(10)
        # for thread in threads:
        #     thread.join()
        
        thread = threading.Thread(target=buffer.recv_and_handle)
        thread.start()
        print(f"接收线程启动完毕")
            
    else:
        buffer.socket.send_string(f"I am rank {rank}")
    print(f"[rank:{rank}] finish send")
        
    
    
if __name__ == "__main__":
    main()