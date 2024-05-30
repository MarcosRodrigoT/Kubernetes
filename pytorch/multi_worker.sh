torchrun \
    --nproc-per-node=2 \
    --nnodes=2 \
    --node-rank=0 \
    --rdzv-id=456 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=tetis:29400 \
    multi_worker_main.py  # Put here additional arguments to your script (if any): --arg1 --arg2...
