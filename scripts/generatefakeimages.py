import argparse
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image,  StableDiffusionXLImg2ImgPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image

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
parser.add_argument('--negative_prompt', type=str,default=None, nargs='+')
parser.add_argument('--seed', type=int,default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
args = parser.parse_args()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

# pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)    

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to(device)


prompt = args.prompt
prompt = ["a couple walking along the riverside, Effiel Tower", " A fairy-tale-like forest, with deer visible in the distance"]
steps = args.steps
negative_prompt = args.negative_prompt

seed = args.seed if args.seed is not None else random.randint(1, 10000000)


# with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
#     out = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed))
with torch.no_grad():
    out = pipe(prompt=prompt,negative_prompt = negative_prompt, generator=set_seed(seed), num_inference_steps=steps)
    out.images[0].save('pics/test2.jpg')
    out.images[1].save('pics/test3.jpg')
