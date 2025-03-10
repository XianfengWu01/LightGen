DATA_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/untar'
CACHE_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/cache_512'
JSON_FILE='/data/xianfeng/code/dataset/dataset.txt'
TEXT_CACHE_PATH='/data/xianfeng/data/text-to-image-2M/data_512_2M/text_cache'

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=29505 main_cache.py --num_workers 0 --batch_size 2048 --mt5_cache_dir /data/xianfeng/code/model/google/flan-t5-xxl \
--img_size 512 --vae_path /data/xianfeng/code/model/stabilityai/stable-diffusion-3.5-large --vae_embed_dim 16 --max_length 128 \
--cached_path ${CACHE_PATH} --data_path ${DATA_PATH} --json_path ${JSON_FILE} --txt_cached_path ${TEXT_CACHE_PATH}