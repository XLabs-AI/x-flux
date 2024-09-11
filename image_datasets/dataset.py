import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='json'):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            img = image_resize(img, self.img_size)
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            if self.caption_type == "json":
                prompt = json.load(open(json_path))['caption']
            else:
                prompt = open(json_path).read()
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
