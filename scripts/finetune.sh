OUTPUT_DIR='fluid_cache_1024'
# DATA_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/text_cache'
DATA_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/untar/data_000047'
JSON_FILE='/data/xianfeng/code/dataset/dataset.txt'
CACHE_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/cache_512'

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 main_fluid.py --num_workers 0 --mt5_cache_dir /data/xianfeng/code/model/google/flan-t5-xxl \
--img_size 1024 --vae_path /data/xianfeng/code/model/stabilityai/stable-diffusion-3.5-large --vae_embed_dim 16 --vae_stride 8 --patch_size 2 --max_length 128 --save_last_freq 1 \
--model fluid_large --diffloss_d 8 --diffloss_w 1536  --epochs 5 --warmup_epochs 0 --batch_size 32 --blr 1.0e-5 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} --grad_checkpointing --finetune --finetune_path /data/xianfeng/code/mar/fluid_cache_512/checkpoint-8.pth \
--data_path ${DATA_PATH} --json_path ${JSON_FILE} --cache_folder ${CACHE_PATH}