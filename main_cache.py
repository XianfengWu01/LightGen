import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import util.misc as misc
# from util.loader import TxtPathDataset # ImageFolderWithFilename

# from models.vae import AutoencoderKL
from models.utils import T5_Embedding
from diffusers.models import AutoencoderKL
from engine_cheat_mar import cache_latents

from util.crop import center_crop_arr
from torch.utils.data import Dataset
from PIL import Image
import json


class ImageCaptionDataset(Dataset):
    def __init__(self, folder_path, num_samples=None, transform=None):
        self.folder_path = folder_path
        self.num_samples = num_samples
        self.transform = transform
        self.file_prefixes = []

        # Collect file prefixes
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                self.file_prefixes.append(os.path.splitext(filename)[0])

        # Random sample
        if self.num_samples is not None and self.num_samples > 0:
            sampled_indices = random.sample(range(len(self.file_prefixes)), self.num_samples)
            self.file_prefixes = [self.file_prefixes[i] for i in sampled_indices]

    def __len__(self):
        return len(self.file_prefixes)

    def __getitem__(self, idx):
        file_prefix = self.file_prefixes[idx]
        image_filename = os.path.join(self.folder_path, f"{file_prefix}.jpg")
        json_filename = os.path.join(self.folder_path, f"{file_prefix}.json")
        # txt_filename = os.path.join(self.folder_path, f"{file_prefix}.txt")

        image = Image.open(image_filename).convert('RGB')
        
        with open(json_filename, 'r') as f:
            caption_data = json.load(f)
            caption = caption_data['prompt']
        # Load text file
        # with open(txt_filename, 'r', encoding='utf-8') as f:
        #     caption = f.read().strip()

        if self.transform:
            image = self.transform(image)

        return image, caption, file_prefix


# Custom Dataset class
class TxtPathDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Custom dataset class to load images from paths specified in a .txt file.
        :param txt_file: Path to the .txt file containing image paths.
        :param root_dir: Root directory of the images (paths in the .txt file are relative to this directory).
        :param transform: Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.relative_paths = []  # List to store the full paths of the images

        # Read image paths from the .txt file
        with open(txt_file, 'r') as file:
            for line in file:
                line = line.strip()  # Remove extra spaces or newlines
                if line:  # Ensure the path is not empty
                    self.relative_paths.append(line)

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.relative_paths)

    def __getitem__(self, idx):
        """
        Load an image based on the index.
        :param idx: Index of the image.
        :return: Image data and its path.
        """
        relative_path = self.relative_paths[idx]
        image_path = os.path.join(self.root_dir, f"{relative_path}.jpg")
        image = Image.open(image_path).convert('RGB')  # Open the image and convert it to RGB mode

        if self.transform:
            image = self.transform(image)  # Apply transformations

        json_filename = os.path.join(self.root_dir, f"{relative_path}.json")
        with open(json_filename, 'r') as f:
            caption_data = json.load(f)
            caption = caption_data['prompt']

        return image, caption, relative_path


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VAE latents', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--json_path', default='main.json', type=str,
                        help='json path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    parser.add_argument('--txt_cached_path', default='', help='path to text cached latents')

    # Text embedding
    parser.add_argument('--mt5_cache_dir', type=str, default='')
    parser.add_argument('--max_length', type=int, default=512)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    # dataset_train = ImageCaptionDataset(folder_path=args.data_path, transform=transform_train)
    dataset_train = TxtPathDataset(txt_file=args.json_path, root_dir=args.data_path, transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )

    # define the vae
    # vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    vae = AutoencoderKL.from_pretrained(os.path.join(args.vae_path, "vae")).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    t5_emb = T5_Embedding(args.mt5_cache_dir, args.mt5_cache_dir, args.max_length).cuda()

    # training
    print(f"Start caching VAE latents")
    start_time = time.time()
    cache_latents(
        vae,
        t5_emb,
        data_loader_train,
        device,
        args=args
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
