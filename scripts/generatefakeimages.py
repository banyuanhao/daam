import argparse
from daam import set_seed
from diffusers import StableDiffusionPipeline
import torch
import random
import cv2

    

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str, default='')
parser.add_argument('--seed', type=int,default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--wandb',action='store_true',help='use wandb')
args = parser.parse_args()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)
prompt = args.prompt
steps = args.steps
negative_prompt = args.negative_prompt
seed = args.seed if args.seed is not None else random.randint(1, 10000000)




with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed))
    print(out.images[0])
    cv2.imwrite('pics/b.png',out.images[0])
