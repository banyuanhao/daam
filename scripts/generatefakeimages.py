from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image,  StableDiffusionXLImg2ImgPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
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

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)  

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")


dataset_path = Path('dataset/ODFN')

prompts_path = os.listdir(dataset_path/'prompts')
prompts_names = os.listdir(dataset_path)
prompts_path = [prompts_path/name for name in prompts_names if name.endswith('.txt')]


seed = [random.randint(1, 10000000) for _ in range(20)]


with torch.no_grad():
    out = pipe(prompt=prompt, generator=set_seed(seed), num_inference_steps=30)
    out.images[0].save('pics/test3.jpg')

