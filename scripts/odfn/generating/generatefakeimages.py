# file to generate images from prompts and seeds

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image,  StableDiffusionXLImg2ImgPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
import os
from utils_odfn import seeds_plus as seeds
from tqdm import tqdm

spilt = 'train'
seeds = seeds[8750:10000]
print('seeds 8750:10000')

class_names = ['baseball_glove.txt']
print(f'classes: {class_names}')


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

# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")


dataset_path = Path(f'dataset/ODFN/version_2')
prompts_path = dataset_path/'prompts_10000_5_5'
prompts_names = os.listdir(prompts_path)
prompts_names = class_names
image_path = dataset_path/spilt/'images'

with torch.no_grad():
    for prompt_name in prompts_names:
        prompt_path = prompts_path/prompt_name
        image_class_path = image_path/prompt_name[0:-4]
        
        if not os.path.exists(image_class_path):
            os.makedirs(image_class_path)
            
        for seed in tqdm(seeds):
            image_seed_path = image_class_path/str(seed)
            if not os.path.exists(image_seed_path):
                os.makedirs(image_seed_path)
            
            with open(prompt_path, 'r') as f:
                prompts = f.read()
                prompts = prompts.split('\n')
                name = prompts[0]
                prompts = prompts[1:]
                
            for k, prompt in enumerate(prompts):
                out = pipe(prompt=prompt, generator=set_seed(seed))
                out.images[0].save(image_seed_path/f'{name}_{seed}_{k}.jpg')