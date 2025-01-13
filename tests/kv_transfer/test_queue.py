import torch  
import queue 

request_queue = queue.Queue()

request_queue.put([0,1])
request_queue.put([0,2])
request_queue.put([0,3])

print(list(request_queue.queue))

pd_pair = request_queue.get()
print(pd_pair)
print(list(request_queue.queue))
