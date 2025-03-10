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

# from models import fluid # video_fluid_arbitrary
from models import fluid_arbitrary as fluid
from models.utils import T5_Embedding
import argparse
from pathlib import Path
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset


def convert_torch_to_int(data):
    if isinstance(data, torch.Tensor):
        return int(data.item())
    elif isinstance(data, list):
        return [convert_torch_to_int(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_torch_to_int(value) for key, value in data.items()}
    else:
        return data


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class GenEvalDataset(Dataset):
    def __init__(self, metadata_file, num_samples=None):

        with open(metadata_file) as fp:
            self.metadatas = [json.loads(line) for line in fp]

        # Random sample
        if num_samples is not None and num_samples > 0:
            sampled_indices = random.sample(range(len(self.file_prefixes)), min(num_samples, len(self.file_prefixes)))
            self.metadatas = [self.metadatas[i] for i in sampled_indices]

    def __len__(self):
        return len(self.metadatas)

    def __getitem__(self, idx):
        metadata = self.metadatas[idx]

        return idx, metadata


def main():
    # init the model architecture
    # img_size = 256
    vae_stride = 8
    patch_size = 2
    diffloss_d = 8
    diffloss_w = 1536
    num_sampling_steps = '100'
    # model_type = "mar_large"
    model_type = 'fluid_large'
    folder_path = "/data/xianfeng/test/dataset/text2image"
    output_folder = "/data/xianfeng/code/geneval/dpo_20epoch"
    batch_size = 1
    num_workers = 0
    output_dir = output_folder
    os.makedirs(output_dir, exist_ok=True)

    # MODEL setting
    # model_checkpoint_path = "/data/xianfeng/code/mar/fluid_cache/checkpoint-95.pth"
    model_checkpoint_path = '/data/xianfeng/code/mar/fluid_dpo/checkpoint-last.pth'
    vae_checkpoint_path = '/data/xianfeng/code/model/stabilityai/stable-diffusion-3.5-large'
    mt5_cache_dir = '/data/xianfeng/code/model/google/flan-t5-xxl'
    max_length = 128

    seed = 42
    num_ar_steps = 64
    cfg_scale = 4.0
    cfg_schedule = "constant"
    temperature = 1.0

    # geneval setting
    metadata_file = '/data/xianfeng/code/geneval/prompts/evaluation_metadata.jsonl'
    n_samples = 4

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

    dataset = GenEvalDataset(metadata_file=metadata_file)
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
    for index, metadata in tqdm(dataloader, disable=dist.get_rank() != 0):
        outpath = os.path.join(output_folder, f"{index.item():0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        # metadata['include'][0]['count'] = metadata['include'][0]['count'].item()
        metadata = convert_torch_to_int(metadata)

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp, indent=4, default=default_dump)
        caption = prompt * n_samples 
        # generate the tokens and images
        with torch.no_grad():
            text_emb = t5_emb(caption)
            with torch.cuda.amp.autocast():
                sampled_tokens = model.sample_tokens(
                    bsz=len(caption), num_iter=num_ar_steps,
                    cfg=cfg_scale, cfg_schedule=cfg_schedule,
                    texts=text_emb, height=512, width=512, # height=256, width=256, # 
                    temperature=temperature, progress=True,
                )

                if vae.config.shift_factor is not None:
                    samples = sampled_tokens / vae.config.scaling_factor + vae.config.shift_factor
                else:
                    samples = sampled_tokens / vae.config.scaling_factor   
                output_samples = vae.decode(samples).sample

        # save the images
        image_path = os.path.join(outpath, "gird.png")
        samples_per_row = n_samples

        save_image(
            output_samples, image_path, nrow=int(samples_per_row), normalize=True, value_range=(-1, 1)
        )
        output_samples = (output_samples / 2 + 0.5).clamp(0, 1)
        output_samples = output_samples.cpu().float()

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        for b_id in range(output_samples.size(0)):
            gen_img = np.round(np.clip(output_samples[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(sample_path, '{}.png'.format(str(b_id).zfill(4))), gen_img)

if __name__ == '__main__':
    main()
