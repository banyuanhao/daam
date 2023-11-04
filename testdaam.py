import argparse
from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str, default='')
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--head_id', type=int, nargs='+', default=None)
parser.add_argument('--layer_id', type=int, nargs='+', default=None)
parser.add_argument('--time_id', type=int, nargs='+', default=None)
parser.add_argument('--factors', type=int, nargs='+', default=None)

parser.add_argument('--words', metavar='S', type=str, nargs='+',
                    help='a string for the string list')

args = parser.parse_args()


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = args.prompt
negative_prompt = args.negative_prompt
seed = set_seed(args.seed)
words = args.words if args.words is not None else []

layer_id = args.layer_id
head_id = args.head_id
time_id = args.time_id
factors = args.factors




#print(type(prompt),type(negative_prompt),type(words),type(seed))

folder_path = Path('experiment')
if not folder_path.is_dir():
    folder_path.mkdir(parents=True, exist_ok=True)
else:
    #raise ValueError(f'Folder {folder_path} already exists!')
    pass
names = os.listdir(folder_path)

i = 1
while(str(i) + '.png' in names):
    i += 1
title = str(i)

with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    with trace(pipe) as tc:
        if len(negative_prompt)> 0:
            out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, generator=seed)
        else:
            out = pipe(prompt, num_inference_steps=50, generator=seed)
        
        heat_map = tc.compute_global_heat_map(factors=factors, head_idx=head_id, layer_idx=layer_id, time_idx=time_id)

#         plt.clf()
#         plt.rcParams.update({'font.size': 16})
#         fig, axs = plt.subplots(1, len(words) + 1, figsize=(5*(len(words)+1), 5+1))

#         for ax in axs:
#             ax.axis('off')

#         axs[0].imshow(out.images[0])

#         for i, word in enumerate(words):
#             heat_map_word = heat_map.compute_word_heat_map(word)
#             heat_map_word.plot_overlay(out.images[0], ax=axs[i+1])

#         plt.subplots_adjust(top=0.98)
#         plt.savefig(f'experiment/{title}.png', bbox_inches='tight')

# print(f"experiment: {title}")
# with open( f'experiment/description.txt', 'a', encoding='utf-8') as file:
#     file.write(f"experiment: {title}\n")
#     file.write(f"prompt: {args.prompt}\n")             
#     file.write(f"negative_prompt: {args.negative_prompt}\n")    
#     file.write(f"words: {words}\n")          
#     file.write(f"seed: {args.seed}\n\n")            
