import torch 
import zmq 
import argparse
from vllm.distributed.kv_transfer.data import PullKey
import time 
import sys 
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--rank', help="rank of process", type=int)
args = parser.parse_args()
rank = args.rank

context = zmq.Context()

# print(f"[rank:{rank}] ")

if rank == 0:
    prefiller = context.socket(zmq.DEALER)
    prefiller_id = f"P-{rank}"
    prefiller.setsockopt(zmq.IDENTITY, prefiller_id.encode())
    prefiller.connect("tcp://localhost:5556")  # 连接到代理的后端
    while True:
        message = prefiller.recv()
        pull_key = pickle.loads(message)    
        print(f"[rank:{rank}] prefiller {prefiller_id} received message: {pull_key.__dict__}")
        
elif rank == 1:
    prefiller = context.socket(zmq.DEALER)
    prefiller_id = f"P-{rank}"
    prefiller.setsockopt(zmq.IDENTITY, prefiller_id.encode())
    prefiller.connect("tcp://localhost:5556")  # 连接到代理的后端
    while True:
        message = prefiller.recv()
        pull_key = pickle.loads(message)
        print(f"[rank:{rank}] prefiller {prefiller_id} received message: {pull_key.__dict__}")
    
elif rank == 2:
    decoder = context.socket(zmq.DEALER)
    decoder.setsockopt(zmq.IDENTITY, f"D-{rank}".encode())
    decoder.connect("tcp://localhost:5555")  # # 连接到代理的前端
    
    for i in range(5):
        pd_pair = [0,rank]
        input_tokens = torch.randn((1,2))
        pull_key = PullKey(pd_pair=pd_pair, input_tokens=input_tokens)
        serialized_pull_key = pickle.dumps(pull_key)
        decoder.send(serialized_pull_key)
        print(f"[rank:{rank}] Sent message: {pull_key.__dict__}")
        time.sleep(0.01)
    
elif rank == 3:
    decoder = context.socket(zmq.DEALER)
    decoder.setsockopt(zmq.IDENTITY, f"D-{rank}".encode())
    decoder.connect("tcp://localhost:5555")  # # 连接到代理的前端
    
    # decoder = context.socket(zmq.DEALER)
    # decoder.connect("tcp://localhost:5555")  # 连接到代理的前端
    for i in range(5):
        pd_pair = [1,rank]
        input_tokens = torch.randn((1,3))
        pull_key = PullKey(pd_pair=pd_pair, input_tokens=input_tokens)
        serialized_pull_key = pickle.dumps(pull_key)
        decoder.send(serialized_pull_key)
        print(f"[rank:{rank}] Sent message: {pull_key.__dict__}")
        time.sleep(0.02)