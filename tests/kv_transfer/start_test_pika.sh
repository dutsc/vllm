file_path="./test_pika.py"

CUDA_VISIBLE_DEVICES=0 python $file_path 0 &
CUDA_VISIBLE_DEVICES=1 python $file_path 1 &
CUDA_VISIBLE_DEVICES=4 python $file_path 2 &
CUDA_VISIBLE_DEVICES=5 python $file_path 3 &

wait