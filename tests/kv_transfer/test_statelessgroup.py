from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclCommunicator, PyNcclPipe
import sys, os
import torch 
from typing import Callable, Dict, Optional, Tuple
device="cuda:0"

def _get_device_send_recv_impl(
        group: StatelessProcessGroup,
    ) -> Tuple[Callable[[torch.Tensor, int], None], Callable[
        [torch.Tensor, int], None]]:

        send: Callable[[torch.Tensor, int], None]
        recv: Callable[[torch.Tensor, int], None]
        comm = PyNcclCommunicator(group, device=0)
        comm.disabled = False
        send, recv = comm.send, comm.recv  # type: ignore
        return send, recv
    
def send_recv(rank,device_send_func,device_recv_func):
    if rank == 0:
        tensor = torch.tensor(1,dtype=torch.int64)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        device_send_func(tensor.to(device),2)
        print(f"[rank:{rank}] finish send tensor:{tensor}")
        
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        device_send_func(tensor.to(device),1)
        print(f"[rank:{rank}] finish send tensor:{tensor}")
    elif rank == 1:
        buffer = torch.empty((1),
                           dtype=torch.int64,
                           device=device)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        print(f"[rank:{rank}] finish recv tensor:{buffer}")
    elif rank == 2:
        buffer = torch.empty((1),
                           dtype=torch.int64,
                           device=device)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        device_recv_func(buffer, 0)
        print(f"[rank:{rank}] finish recv tensor:{buffer}")
def main():
    rank = int(sys.argv[1])
    port = 12344
    print(f"[rank:{rank}]")
    print(f"[port:{port}]")
    
    group = StatelessProcessGroup.create(
        host="127.0.0.1",
        port=port,
        rank=rank,
        world_size=4
    )
    group.barrier()
    print(f"[rank{rank}] finish create group")
    impl = _get_device_send_recv_impl(group=group)
    device_send_func, device_recv_func = impl
    print(f"[rank{rank}] finish init device_send_func, device_recv_func")
    send_recv(rank, device_send_func, device_recv_func)
if __name__ == "__main__":
    main()