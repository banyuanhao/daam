# mean activation value of the feature maps

import argparse
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
# from models.diffuserpipeline import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import json
#
parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str, default='')
parser.add_argument('--seed', type=int,nargs='+', default=[0])
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--group', type=str, required=True, 
                    choices=['noun_removal', 'noun_style', 'noun_general', 'noun_abstract',
                             'adj_noun','adj_noun_0','adj_noun_1',
                             'adj_general','adj_specific',
                             'steps_adj','steps_noun'])
parser.add_argument('--token',type=int, default=None)
args = parser.parse_args()


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = args.prompt
negative_prompt = args.negative_prompt
seeds = args.seed
steps = args.steps


ratio_list = []

def int_token(negative_prompt, int):
    token = negative_prompt.split(' ')
    if int == 0 and len(token) == 2:
        return token[0]
    elif int == 0 and len(token) == 3:
        return ' '.join(token[:2])
    else:
        return token[-1]

for seed in iter(seeds):
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            
            out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed),output_type='latent')
            if args.group != 'adj_noun_0' and args.group != 'adj_noun_1':
                pos, neg = tc.compute_activation_ratio()
            else:
                pos, neg = tc.compute_activation_ratio_spec(token = int_token(negative_prompt, args.token))
            ratio = [neg[i]/pos[i] for i in range(len(pos))]
            ratio_list.append(ratio)

    
import os

if os.path.exists(f'wrapupdata/rebuttal/generalize/{args.group}.json'):
    with open(f'wrapupdata/rebuttal/generalize/{args.group}.json','r') as f:
        data = json.load(f)
else:
    data = []
    
data.append({'prompt':prompt, 'negative_prompt':negative_prompt, 'seeds':seeds, 'steps':steps, 'ratio':ratio_list, 'group':args.group, 'token':args.token})

with open(f'wrapupdata/rebuttal/generalize/{args.group}.json', 'w') as f:
    json.dump(data, f)