import zmq
import pickle
from vllm.distributed import get_kv_transfer_group

# kv_transfer_config = get_kv_transfer_group().config.kv_transfer_config
# prefiller_port = kv_transfer_config.prefiller_port
# decoder_port = kv_transfer_config.decoder_port

prefiller_port = 12312
decoder_port = 12311

context = zmq.Context()
frontend = context.socket(zmq.ROUTER)
frontend.bind(f"tcp://*:{decoder_port}")
backend = context.socket(zmq.ROUTER)
backend.bind(f"tcp://*:{prefiller_port}")

print("Proxy started. Waiting for messages...")

while True:
    identity = frontend.recv()  # 接收身份信息
    message = frontend.recv()   # 接收实际消息
    
    pull_key = pickle.loads(message)
    pd_pair = pull_key.pd_pair

    # print(f"Received message from {identity}: {pd_pair}") 
    p_rank = pd_pair[0]
    
    target_p_id = f"P-{p_rank}".encode()

    # 将消息发送到指定的P
    backend.send_multipart([target_p_id,message])

