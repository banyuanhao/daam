import argparse
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from tqdm import tqdm

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
    

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str,default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--negative_time', type=int, nargs = '+', default=None)
args = parser.parse_args()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

pipe = StableDiffusionPipelineForNegativePrompts.from_pretrained(model_id, use_auth_token=True).to(device)

prompt = args.prompt
steps = args.steps
negative_prompt = args.negative_prompt
negative_time = args.negative_time

for seed in tqdm(range(1000)):
    with torch.no_grad():
        out = pipe(prompt=prompt,negative_prompt = negative_prompt, generator=set_seed(seed), num_inference_steps=steps, negative_time=negative_time)
        out.images[0].save(f'pics/removing/glasses/negative_6_12/{seed}.png')