OUTPUT_DIR='fluid_cache'
DATA_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/untar'
JSON_FILE='/data/xianfeng/code/dataset/dataset.txt'
CACHE_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/cache_256'

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=29501 main_cheat_mar.py --num_workers 0 --mt5_cache_dir /data/xianfeng/code/model/google/flan-t5-xxl \
--img_size 256 --vae_path /data/xianfeng/code/model/stabilityai/stable-diffusion-3.5-large --vae_embed_dim 16 --vae_stride 8 --patch_size 2 --max_length 128 --save_last_freq 5 \
--model mar_large --diffloss_d 8 --diffloss_w 1536  --epochs 1000 --warmup_epochs 100 --batch_size 256 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} --use_cached --grad_checkpointing \
--data_path ${DATA_PATH} --json_path ${JSON_FILE} --cache_folder ${CACHE_PATH}