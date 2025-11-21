from transformers import ViTModel, ViTImageProcessor, ViTFeatureExtractor
from PIL import Image
import requests
import torch
from torch import nn
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
# import torchvision.models.resnet as models
from PIL import Image
import h5py
import numpy as np
import scipy.io as sio
import pickle
import os
from tqdm import tqdm
from thop import profile

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
#
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
# inputs = processor(images=image, return_tensors="pt")
#
# outputs = model(**inputs)
# features = outputs.last_hidden_state
#
# print(outputs.pooler_output.size())

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = model.to(device)

flops, params = profile(model, inputs=(torch.rand(1, 3, 224, 224).to('cuda:1')), verbose=False)
# flops_str, params_str = clever_format([flops, params], "G")
print(f"GFLOPS: {flops / 1e9}")
print(f"Params: {params / 1e6}")


class CustomedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir , file_paths, transform=None):
        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        image_file = os.path.join(self.img_dir, '/'.join(image_file.replace('//', '/').split('/')[6:]))
        image = Image.open(image_file).convert('RGB')

        return image

def collate_func(batch):
    pictures = []
    for items in batch:
        pictures.append(items)
    return processor(images=pictures, return_tensors="pt")


def extract_features(model, img_dir, file_path, output_dir):
    dataset = CustomedDataset(img_dir, file_path)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=64, shuffle=False,
                                                 num_workers=4, collate_fn=collate_func)
    all_features = []
    for imgs in tqdm(dataset_loader):
        images = imgs.to(device)
        features = model(**images).pooler_output
        all_features.append(features.detach().cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)
    sio.savemat(os.path.join(output_dir, 'vit_features.mat'), {'features': all_features})


if __name__ == '__main__':
    extract_features(model, img_dir='./dataset/AWA2/',
                     file_path='./dataset/AWA2/res101.mat',
                     output_dir='./dataset/AWA2/')
    vit_feat = sio.loadmat('./dataset/AWA2/vit_features.mat')
    print(vit_feat['features'].shape)