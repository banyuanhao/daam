# generate noises fed into the detector, one seed one noise
# npz format, read using LoadImageFromNPY

import numpy as np
from diffusers import StableDiffusionPipeline
from utils_odfn import seeds_spilt, set_seed,seeds_plus_spilt
from pathlib import Path
from PIL import Image

spilt = 'val'
version = 'version_2'

for spilt in ['train', 'val', 'test']:
    seeds_sub = seeds_plus_spilt(spilt)
    model_id = 'stabilityai/stable-diffusion-2-base'
    device = 'cuda'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)  

    dataset_path = Path(f'dataset/ODFN/{version}/{spilt}/noises_png')

    for seed in seeds_sub:
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True)
        prompt = "none"
        latent = pipe.get_latents(prompt=prompt, generator=set_seed(seed))
        latent = latent[0].cpu().numpy().transpose(1,2,0)
            
        # LoadImagesFromFile
        # scale latent to [0,1]
        img = latent
        img = img - np.min(img)
        img = img / np.max(img)
        #compute the mean of the latent
        img = np.mean(img, axis=2,keepdims = True)
        img = np.repeat(img, 3, axis=2)
        
        # save as png
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(dataset_path / f'{seed}.png')
        
        
        # LoadImageFromNPY
        # save latent as npy
        # np.save(dataset_path / f'{seed}.npy', latent)
        
        # LoadMultipleImagesFromFile
        # for i in range(4):
        #     img = latent[0][i].cpu().numpy()
        #     print(np.max(img))
        #     print(np.min(img))  
        #     mmcv.imwrite(img, seed_path / f'{i}.png')
            
            
        