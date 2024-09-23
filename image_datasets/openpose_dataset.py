import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

class OpenPoseImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.img_dir = img_dir
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if ('.jpg' in i or '.png' in i) and not i.endswith('_pose.jpg') and not i.endswith('_pose.png')]
        self.images.sort()
        self.img_size = img_size

        print('OpenPoseImageDataset: ', len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            json_path = self.images[idx].split('.')[0] + '.json'
            json_data = json.load(open(json_path))

            img = Image.open(self.images[idx])
            img = c_crop(img)
            img = img.resize((self.img_size, self.img_size))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)

            hint_path = os.path.join(self.img_dir, json_data['conditioning_image'])
            hint = Image.open(hint_path)
            hint = c_crop(hint)
            hint = hint.resize((self.img_size, self.img_size))
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            
            prompt = json_data['caption']
            return img, hint, prompt

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def openpose_dataset_loader(train_batch_size, num_workers, **args):
    dataset = OpenPoseImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
