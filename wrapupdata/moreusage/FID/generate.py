# mean activation value of the feature maps
import argparse
import os
from diffusers import StableDiffusionPipeline
import torch
from daam import set_seed
import json
with open('/home/banyh2000/diffusion/daam/wrapupdata/moreusage/FID/seeds.json', 'r') as f:
    seeds = json.load(f)

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = "a cat on the sofa"
negative_time = 31
# negative_prompt = ['blurry',
#               'distorted',
#               'unfocused',
#               'deformed',
#               'disfigured',
#               'ugly']
negative_prompt = ['blurry',
                   'distorted',
                   'uncute']


path = '/home/banyh2000/diffusion/daam/wrapupdata/moreusage/FID/cat/generated'
path = os.path.join(path, str(negative_time))
os.makedirs(path, exist_ok=True)

with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    for i in range(1000):
        for j in range(len(negative_prompt)):
            out = pipe.alternating(prompt, negative_prompt=negative_prompt[j], num_inference_steps=30, generator=set_seed(seeds[i]),negative_time = negative_time)
            out.images[0].save(f'{path}/{i}_{j}.png')