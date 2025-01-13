import torch 
import pika 
import os, sys 
from typing import Deque, List, Optional
import threading 
from collections import deque


class SimpleBuffer():
    def __init__(self, rank):
        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_lock = threading.Lock()
        self.kv_rank = rank
        user_info = pika.PlainCredentials('guest', 'guest')#用户名和密码
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1', 5672, '/', user_info))#连接服务器上的RabbitMQ服务

        # 创建一个channel
        self.channel = self.connection.channel()

        if rank >= 1:
            pair_hash = self.hash_rank([0,rank])
            # 如果指定的queue不存在，则会创建一个queue，如果已经存在 则不会做其他动作，官方推荐，每次使用时都可以加上这句
            self.channel.queue_declare(queue=pair_hash)

    
    def hash_rank(self, pd_pair: List[int]):
        p_rank = pd_pair[0]
        d_rank = pd_pair[1]
        return p_rank * 100 + d_rank * 10
    
    
def main():
    rank = sys.argv[1]
    print(f"[rank:{rank}]")
    buffer = SimpleBuffer(rank)
    
if __name__ == "__main__":
    main()