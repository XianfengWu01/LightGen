import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from datetime import timedelta
import json

import torch
import torch.distributed as dist
from torchvision.utils import save_image
from torch.utils.data import DataLoader, DistributedSampler

from models import fluid_arbitrary as fluid # video_fluid_arbitrary
from models.utils import T5_Embedding
import argparse
from pathlib import Path
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_json_entry(entry, path):
    with open(path, 'a') as outfile:
        json.dump(entry, outfile, separators=(',', ':'), default=default_dump)
        outfile.write('\n')


def convert_json_line_to_general(input_json):
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        with open(input_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=default_dump)
    except json.JSONDecodeError as e:
        print("this filtering process completed")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


class DPOCaptionDataset(Dataset):
    def __init__(self, input_json_file):
        with open(input_json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        file_prefix = os.path.splitext(os.path.basename(data_info['image_path']))[0]
        caption = data_info['caption']
        image_path = data_info['image_path']

        return caption, file_prefix, image_path


def main():
    # init the model architecture
    # img_size = 256
    vae_stride = 8
    patch_size = 2
    diffloss_d = 8
    diffloss_w = 1536
    num_sampling_steps = '100'
    model_type = "fluid_large"
    input_json_file = '/data/xianfeng/data/captioned_data_dpo.json'
    output_json_file = '/data/xianfeng/data/dpo_data.json'
    output_folder = "/data/xianfeng/data/laion-600k-aesthetic-6.5plus-768-32k/negative_sample"
    batch_size = 128
    num_workers = 0
    os.makedirs(output_folder, exist_ok=True)

    # MODEL setting
    model_checkpoint_path = "fluid_cache_512/checkpoint-last.pth"
    vae_checkpoint_path = '/data/xianfeng/code/model/stabilityai/stable-diffusion-3.5-large'
    mt5_cache_dir = '/data/xianfeng/code/model/google/flan-t5-xxl'
    max_length = 128

    seed = 42
    num_ar_steps = 64
    cfg_scale = 4.0
    cfg_schedule = "constant"
    temperature = 1.0

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = fluid.__dict__[model_type](
        vae_stride=vae_stride,
        patch_size=patch_size,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        text_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        diffloss_d=diffloss_d,
        diffloss_w=diffloss_w,
        max_length=128,
        num_sampling_steps=num_sampling_steps,
    ).cuda()

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.cuda()

    # download and load the vae
    
    vae = AutoencoderKL.from_pretrained(os.path.join(vae_checkpoint_path, "vae")).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    t5_emb = T5_Embedding(mt5_cache_dir, mt5_cache_dir, max_length).cuda()

    # set up user-specified or default values for generation
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = DPOCaptionDataset(input_json_file=input_json_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        ),
    )

    model.eval()
    for captions, file_prefixs, image_paths in tqdm(dataloader, disable=dist.get_rank() != 0):
        # generate the tokens and images
        with torch.no_grad():
            text_emb = t5_emb(captions)
            with torch.cuda.amp.autocast():
                sampled_tokens = model.sample_tokens(
                    bsz=len(captions), num_iter=num_ar_steps,
                    cfg=cfg_scale, cfg_schedule=cfg_schedule,
                    texts=text_emb, height=512, width=512, 
                    temperature=temperature, progress=True,
                )

                if vae.config.shift_factor is not None:
                    samples = sampled_tokens / vae.config.scaling_factor + vae.config.shift_factor
                else:
                    samples = sampled_tokens / vae.config.scaling_factor   
                output_samples = vae.decode(samples).sample

        output_samples = (output_samples / 2 + 0.5).clamp(0, 1)
        output_samples = output_samples.cpu().float()

        for b_id in range(output_samples.size(0)):
            gen_img = np.round(np.clip(output_samples[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(output_folder, f'{file_prefixs[b_id]}.jpg'), gen_img)

            item = {"pos_image_path": image_paths[b_id], "neg_image_path": os.path.join(output_folder, f'{file_prefixs[b_id]}.jpg')}
            item["caption"] = captions[b_id]
            save_json_entry(item, output_json_file)
    
    dist.barrier()
    torch.cuda.empty_cache()

    # print(f"Processing completed. Outputs saved to {args.output_json_file} and images saved to {args.output_path}")
    dist.barrier(device_ids=[local_rank])
    convert_json_line_to_general(output_json_file)
    print(f"Processing completed. Outputs saved to {output_json_file}. ")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
