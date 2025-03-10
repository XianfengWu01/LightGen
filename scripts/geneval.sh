# CUDA_VISIBLE_DEVICES=6,7 
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port 29507 -m geneval