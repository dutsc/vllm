file_path="./test_router.py"
proxy_path="./zmq_proxy.py"

python $proxy_path &
python $file_path --rank 0 &
python $file_path --rank 1 &
python $file_path --rank 2 &
python $file_path --rank 3 &