import argparse
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
transform = transforms.ToTensor()

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
parser.add_argument('--seeds', type=int, nargs='+', default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--estimated_time', type=int, nargs = '+', default=None)
parser.add_argument('--negative_time', type=int, nargs = '+', default=None)
args = parser.parse_args()

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = 'cuda'


pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

prompt = args.prompt
steps = args.steps
negative_prompt = args.negative_prompt
seeds = args.seeds
prompt = "a photo of an astronaut riding a horse on mars"

for seed in seeds:
    with torch.no_grad():
        out = pipe(prompt=prompt)
        plt.imshow(out.images[0])
        print(out.images[0])
        # convert Image to tensor
        image = transform(out.images[0])
        print(image.shape)
        print(image)
        print(torch.min(image))
        print(torch.max(image))
        plt.imsave(f"pics/{seed}.png", out.images[0])