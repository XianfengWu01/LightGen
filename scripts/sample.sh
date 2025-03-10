# CUDA_VISIBLE_DEVICES=0,1,3,4,5,7 
# torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 -m dpo_sample
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=29505 -m sample
# torchrun --nproc_per_node=6 --nnodes=1 --node_rank=0 -m dpo_sample