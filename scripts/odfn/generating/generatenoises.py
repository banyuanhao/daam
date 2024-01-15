# generate noises fed into the detector, one seed one noise
# npz format, read using LoadImageFromNPY

import numpy as np
from diffusers import StableDiffusionPipeline
from utils_odfn import seeds
import torch
import random
from pathlib import Path
from typing import TypeVar
T = TypeVar('T')
import mmcv
import os

def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)  

spilt = 'val'
if spilt == 'test':
    seeds_sub = seeds[90:95]
elif spilt == 'val':
    seeds_sub = seeds[80:85]
elif spilt == 'train':
    seeds_sub = seeds[:40]
else:
    raise ValueError('category error')

dataset_path = Path(f'dataset/ODFN/{spilt}/noises')

for seed in seeds:
    prompt = "chushu"
    latent = pipe.get_latents(prompt=prompt, generator=set_seed(seed))
    latent = latent[0].cpu().numpy().transpose(1,2,0)
    # save latent as npy
    np.save(dataset_path / f'{seed}.npy', latent)
    
    
    # LoadMultipleImagesFromFile
    # for i in range(4):
    #     img = latent[0][i].cpu().numpy()
    #     print(np.max(img))
    #     print(np.min(img))  
    #     mmcv.imwrite(img, seed_path / f'{i}.png')
        
        
    