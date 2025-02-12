import torch 
import queue 
from typing import List 

producer_num = 1
consumer_num = 3
kv_rank = 0

def hash_rank(pd_pair: List[int]):
        p_rank = pd_pair[0]
        d_rank = pd_pair[1]
        return p_rank * 100 + d_rank * 10
    
    
queues = {
    hash_rank([kv_rank,x]):queue.Queue() 
    for x in range(producer_num, producer_num + consumer_num)
}
print(queues.keys())
pd_pair_list = [[kv_rank,x] for x in range(producer_num, producer_num + consumer_num)]
print(pd_pair_list)
