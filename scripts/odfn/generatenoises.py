# generate noises fed into the detector, one seed one noise
# npz format, read using LoadImageFromNPY

import numpy as np
from diffusers import StableDiffusionPipeline
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

seeds = [3502948, 2414292, 4013215, 7661395, 2728259, 7977675, 6926097, 8223344, 4338686, 2630916, 3548081, 3710422, 2285361, 9638421, 2837631, 5468982, 7955021, 7197637, 4206420, 1347815, 2957833, 3326072, 1813088, 7965829, 4708029, 452169, 1107126, 8388604, 9481161, 8020003, 2225075, 1440263, 29403, 7099996, 7851895, 1106978, 4053385, 6882390, 3322966, 3668830, 8613167, 1315399, 3121499, 900759, 7739336, 1464588, 1144945, 39451, 3131354, 6971254, 1088493, 1700896, 3760774, 3410488, 3129936, 5309498, 3698823, 5970284, 2569054, 8264031, 8663422, 5174978, 4041203, 1690212, 7695658, 4857840, 4395970, 2970532, 1313178, 7409679, 1242182, 6902329, 4582656, 4123976, 8158709, 3033046, 1634920, 6750562, 6337306, 8317766, 1618731, 1518909, 4798495, 2620399, 2423703, 7285262, 180696, 8432894, 3157912, 7890161, 5509442, 6216034, 7431925, 7774348, 6443781, 6142998, 3686770, 8916284, 9406101, 7637527]
#seeds = seeds[:40]
seeds = seeds[80:85]
#seeds = seeds[90:95]

dataset_path = Path('dataset/ODFN/val/noises')

for seed in seeds:
    prompt = "chushu"
    latent = pipe.get_latents(prompt=prompt, generator=set_seed(seed))
    latent = latent[0].cpu().numpy().transpose(1,2,0)
    print(latent.shape)
    # save latent as npz
    np.savez(dataset_path / f'{seed}.npz', latent)
    # for i in range(4):
    #     img = latent[0][i].cpu().numpy()
    #     print(np.max(img))
    #     print(np.min(img))  
    #     mmcv.imwrite(img, seed_path / f'{i}.png')
        
# for seed in seeds:
#     read_path = dataset_path / str(seed)
#     for i in range(4):
#         img = mmcv.imread(read_path / f'{i}.png')
#         print(np.max(img))
#         print(np.min(img))
#         print(img.shape)
        
    