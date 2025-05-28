#!/bin/bash

# 使用ps -aux | grep vllm过滤出含vllm的进程，并获取其pid
pids=$(ps -aux | grep vllm | grep -v grep | awk '{print $2}')
pkill -9 python

# 判断是否有匹配的进程
if [ -z "$pids" ]; then
    echo "未找到含vllm的进程"
else
    # 遍历所有匹配的pid，并使用kill -9杀死这些进程
    for pid in $pids
    do
        echo "正在杀死进程：$pid"
        kill -9 $pid
    done
    echo "含vllm的进程已全部被杀死"
fi

pids=$(ps -aux | grep zmq | grep -v grep | awk '{print $2}')
pkill -9 python

# 判断是否有匹配的进程
if [ -z "$pids" ]; then
    echo "未找到含zmq的进程"
else
    # 遍历所有匹配的pid，并使用kill -9杀死这些进程
    for pid in $pids
    do
        echo "正在杀死进程：$pid"
        kill -9 $pid
    done
    echo "含zmq的进程已全部被杀死"
fi