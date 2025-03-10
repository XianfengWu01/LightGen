import os
import numpy as np
import json
from PIL import Image

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target


class LAIONJSONDataset(Dataset):
    def __init__(self, json_file, images_dir, transform=None, num_sample=0):
        with open(json_file, 'r') as f:
            data = json.load(f)
        if num_sample > 0:
            data = data[:num_sample]
        self.data = data
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        image_path = os.path.join(self.images_dir, data_info['image_path']) 
        image = Image.open(image_path).convert('RGB')

        caption = data_info['caption']
        
        if self.transform:
            image = self.transform(image)

        return image, caption


class LAION(Dataset):
    def __init__(self, json_file, images_dir, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.data = data
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        relative_path = data_info['image_path'] + '.jpg'
        caption_path = data_info['image_path'] + '.txt'
        image_path = os.path.join(self.images_dir, relative_path) 
        image = Image.open(image_path).convert('RGB')

        txt_path = os.path.join(self.images_dir, caption_path) 
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        if self.transform:
            image = self.transform(image)

        return image, caption


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

        return image, caption


class CacheCaptionDataset(Dataset):
    def __init__(self, folder_path, cache_folder, num_samples=None):
        self.folder_path = folder_path
        self.cache_folder = cache_folder
        self.num_samples = num_samples
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
        cache_filename = os.path.join(self.cache_folder, f"{file_prefix}.npz")
        json_filename = os.path.join(self.folder_path, f"{file_prefix}.json")
        
        data = np.load(cache_filename)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']
        
        with open(json_filename, 'r') as f:
            caption_data = json.load(f)
            caption = caption_data['prompt']

        return moments, caption


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

        return image, caption


class TxtCacheDataset(Dataset):
    def __init__(self, txt_file, root_dir, cache_folder, transform=None):
        self.root_dir = root_dir
        self.cache_folder = cache_folder
        self.transform = transform
        self.relative_paths = []  # List to store the full paths of the images

        # Read image paths from the .txt file
        with open(txt_file, 'r') as file:
            for line in file:
                line = line.strip()  # Remove extra spaces or newlines
                if line:  # Ensure the path is not empty
                    self.relative_paths.append(line)

    def __len__(self):
        return len(self.relative_paths)

    def __getitem__(self, idx):
        relative_path = self.relative_paths[idx]
        cache_path = os.path.join(self.cache_folder, f"{relative_path}.npz")
        data = np.load(cache_path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        json_filename = os.path.join(self.root_dir, f"{relative_path}.json")
        with open(json_filename, 'r') as f:
            caption_data = json.load(f)
            caption = caption_data['prompt']

        return moments, caption


class CacheDataset(Dataset):
    def __init__(self, txt_file, cache_folder, text_folder):
        self.text_folder = text_folder
        self.cache_folder = cache_folder
        self.relative_paths = []  # List to store the full paths of the images

        # Read image paths from the .txt file
        with open(txt_file, 'r') as file:
            for line in file:
                line = line.strip()  # Remove extra spaces or newlines
                if line:  # Ensure the path is not empty
                    self.relative_paths.append(line)

    def __len__(self):
        return len(self.relative_paths)

    def __getitem__(self, idx):
        relative_path = self.relative_paths[idx]
        cache_path = os.path.join(self.cache_folder, f"{relative_path}.npz")
        data = np.load(cache_path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        text_filename = os.path.join(self.text_folder, f"{relative_path}.npz")
        text_embedding = np.load(text_filename)['text_emb']

        return moments, text_embedding