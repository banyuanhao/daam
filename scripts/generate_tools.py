from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
import matplotlib.pyplot as plt

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

pipe = StableDiffusionPipelineForNegativePrompts.from_pretrained(model_id, use_auth_token=True).to(device)

def generate_image(prompt, negative_prompt, steps, seed, negative_time):
    with torch.no_grad():
        output = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed), negative_time=negative_time)
    image = output.images[0]
    return image



