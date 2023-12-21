import argparse
from daam import set_seed
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import random
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str, default='')
parser.add_argument('--seed', type=int,default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
args = parser.parse_args()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to(device)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to(device)


prompt = args.prompt
steps = args.steps
negative_prompt = args.negative_prompt
seed = args.seed if args.seed is not None else random.randint(1, 10000000)


with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    out = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed))
    print(out.images[0])
    cv2.imwrite('pics/b.png',out.images[0])
