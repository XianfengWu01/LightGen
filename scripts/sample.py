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
import pandas as pd
from models.utils import T5_Embedding
import argparse
from pathlib import Path
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset


# class CaptionDataset(Dataset):
#     def __init__(self, folder_path, num_samples=None):
#         self.folder_path = folder_path
#         self.num_samples = num_samples
#         self.file_prefixes = []

#         # Collect file prefixes
#         for filename in os.listdir(folder_path):
#             if filename.endswith('.jpg'):
#                 self.file_prefixes.append(os.path.splitext(filename)[0])

#         # Random sample
#         if self.num_samples is not None and self.num_samples > 0:
#             sampled_indices = random.sample(range(len(self.file_prefixes)), self.num_samples)
#             self.file_prefixes = [self.file_prefixes[i] for i in sampled_indices]

#     def __len__(self):
#         return len(self.file_prefixes)

#     def __getitem__(self, idx):
#         file_prefix = self.file_prefixes[idx]
#         json_filename = os.path.join(self.folder_path, f"{file_prefix}.json")
        
#         with open(json_filename, 'r') as f:
#             caption_data = json.load(f)
#             caption = caption_data['prompt']

#         return caption, file_prefix


class CaptionDataset(Dataset):
    def __init__(self, folder_path, output_folder, num_samples=None):
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.num_samples = num_samples
        self.file_prefixes = []

        # Collect file prefixes
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                file_prefix = os.path.splitext(filename)[0]
                output_file = os.path.join(output_folder, f"{file_prefix}.npz")
                
                # Check if the output file already exists
                if not os.path.exists(output_file):
                    self.file_prefixes.append(file_prefix)

        # Random sample
        if self.num_samples is not None and self.num_samples > 0:
            sampled_indices = random.sample(range(len(self.file_prefixes)), min(self.num_samples, len(self.file_prefixes)))
            self.file_prefixes = [self.file_prefixes[i] for i in sampled_indices]

    def __len__(self):
        return len(self.file_prefixes)

    def __getitem__(self, idx):
        file_prefix = self.file_prefixes[idx]
        json_filename = os.path.join(self.folder_path, f"{file_prefix}.json")
        
        with open(json_filename, 'r') as f:
            caption_data = json.load(f)
            caption = caption_data['prompt']

        return caption, file_prefix


def main():
    # init the model architecture
    # img_size = 256
    vae_stride = 8
    patch_size = 2
    diffloss_d = 8
    diffloss_w = 1536
    num_sampling_steps = '1000'
    model_type = "fluid_large"
    # folder_path = "/data/xianfeng/test/dataset/text2image"
    filepath = '/data/xianfeng/code/text2image-benchmark/MS-COCO_val2014_30k_captions.csv'
    output_folder = "/data/xianfeng/code/text2image-benchmark/cache_512_1000"
    batch_size = 64
    num_workers = 0
    # output_dir = "cache_100k"
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

    dataset = CaptionDataset(folder_path=folder_path, output_folder=output_folder)
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
    for caption, file_prefix in tqdm(dataloader, disable=dist.get_rank() != 0):
        # generate the tokens and images
        with torch.no_grad():
            text_emb = t5_emb(caption)
            with torch.cuda.amp.autocast():
                sampled_tokens = model.sample_tokens(
                    bsz=len(caption), num_iter=num_ar_steps,
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
        
        # save_folder = output_folder

        for b_id in range(output_samples.size(0)):
            gen_img = np.round(np.clip(output_samples[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(output_folder, f'{file_prefix[b_id]}.jpg'), gen_img)
            # np.savez(os.path.join(save_folder, file_prefix[b_id] + '.npz'), feature=samples[b_id].cpu().numpy())

if __name__ == '__main__':
    main()
