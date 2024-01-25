# generate noises fed into the detector, one seed one noise
# npz format, read using LoadImageFromNPY

import numpy as np
from diffusers import StableDiffusionPipeline
from utils_odfn import seeds_spilt, set_seed,seeds_plus_spilt
from pathlib import Path

spilt = 'val'
version = 'version_2'

for spilt in ['train', 'val', 'test']:
    seeds_sub = seeds_plus_spilt(spilt)
    model_id = 'stabilityai/stable-diffusion-2-base'
    device = 'cuda'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)  

    dataset_path = Path(f'dataset/ODFN/{version}/{spilt}/noises')

    for seed in seeds_sub:
        prompt = "chushu"
        latent = pipe.get_latents(prompt=prompt, generator=set_seed(seed))
        latent = latent[0].cpu().numpy().transpose(1,2,0)
        # save latent as npy
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True)
        np.save(dataset_path / f'{seed}.npy', latent)
        
        # LoadMultipleImagesFromFile
        # for i in range(4):
        #     img = latent[0][i].cpu().numpy()
        #     print(np.max(img))
        #     print(np.min(img))  
        #     mmcv.imwrite(img, seed_path / f'{i}.png')
            
            
        