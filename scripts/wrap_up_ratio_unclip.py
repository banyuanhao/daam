import argparse
import sys
sys.path.append('~/diffusion/daam')
from daam import trace
from daam import set_seed
import torch
from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from matplotlib import pyplot as plt

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
                             'steps_adj','steps_noun','unclip_adj','unclip_noun'])
parser.add_argument('--token',type=int, default=None)
args = parser.parse_args()

prior_model_id = "kakaobrain/karlo-v1-alpha"
data_type = torch.float16
prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)
prior_text_model_id = "openai/clip-vit-large-patch14"
prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)
stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

pipe = StableUnCLIPPipeline.from_pretrained(
    stable_unclip_model_id,
    torch_dtype=data_type,
    variant="fp16",
    prior_tokenizer=prior_tokenizer,
    prior_text_encoder=prior_text_model,
    prior=prior,
    prior_scheduler=prior_scheduler,
)

pipe = pipe.to("cuda")

prompt = args.prompt
negative_prompt = args.negative_prompt
seeds = args.seed
steps = args.steps

ratio_list = []

# with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
# with torch.no_grad():
#     with trace(pipe) as tc:
#         out = pipe(prompt, num_inference_steps=30, negative_prompt=negative_prompt)
#         pos, neg = tc.compute_activation_ratio()
#         print(neg)
#         heat_map = tc.compute_global_heat_map()
#         heat_map = heat_map.compute_word_heat_map("street")
#         heat_map.plot_overlay(out.images[0],ax = plt.gca())
#         heat_ = tc.return_heat_map()
        
for seed in iter(seeds):
    with torch.no_grad():
        with trace(pipe) as tc:
            
            out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed),output_type='latent')
            pos, neg = tc.compute_activation_ratio()
            ratio = [neg[i]/pos[i] for i in range(len(pos))]
            ratio_list.append(ratio)


import os
import json

if os.path.exists(f'wrapupdata/rebuttal/generalize/{args.group}.json'):
    with open(f'wrapupdata/rebuttal/generalize/{args.group}.json','r') as f:
        data = json.load(f)
else:
    data = []
    
data.append({'prompt':prompt, 'negative_prompt':negative_prompt, 'seeds':seeds, 'steps':steps, 'ratio':ratio_list, 'group':args.group, 'token':args.token})

with open(f'wrapupdata/rebuttal/generalize/{args.group}.json', 'w') as f:
    json.dump(data, f)