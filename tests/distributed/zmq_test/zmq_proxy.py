import zmq
import pickle

# 创建 ZeroMQ 上下文
context = zmq.Context()

# 创建前端 ROUTER 套接字  连接到Decoder
frontend = context.socket(zmq.ROUTER) 
frontend.bind("tcp://*:5555")  # 绑定到端口 5555

# 创建后端 DEALER 套接字  连接到Prefiller
backend = context.socket(zmq.ROUTER)
backend.bind("tcp://*:5556")  # 绑定到端口 5556

print("Proxy started. Waiting for messages...")

while True:
    frames = frontend.recv_multipart()
    sender_identity = frames[0]
    message = frames[-1]
    # identity = frontend.recv()  # 接收身份信息
    # message = frontend.recv()   # 接收实际消息
    
    pull_key = pickle.loads(message)
    pd_pair = pull_key.pd_pair

    p_rank = pd_pair[0]
    
    target_p_id = f"P-{p_rank}".encode()
    
    print(f"[Proxy] Routing message from {sender_identity} to {target_p_id}")
    print(f"        pd_pair={pull_key.pd_pair}, data_shape={pull_key.input_tokens.shape}")

    # 将消息发送到指定的P
    backend.send_multipart([target_p_id,message])

