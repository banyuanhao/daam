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
parser.add_argument('--experiment_id', type=str, required=True)

parser.add_argument('--words', metavar='S', type=str, nargs='+',
                    help='a string for the string list')


args = parser.parse_args()


model_id = 'stabilityai/stable-diffusion-2'
device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = args.prompt
negative_prompt = args.negative_prompt
print(prompt)
seed = set_seed(args.seed)
words = args.words if args.words is not None else []
words = [word.replace('_', ' ') for word in words]

layer_id = args.layer_id
factors = args.factors
time_id = args.time_id
head_id = args.head_id
experiment_id = args.experiment_id

layer_id = None if layer_id is None else list(range(layer_id[0], layer_id[1] + 1))
time_id = None if time_id is None else list(range(time_id[0], time_id[1] + 1))

#print(type(prompt),type(negative_prompt),type(words),type(seed))

folder_path = Path('experiment') /experiment_id

if not folder_path.is_dir():
    folder_path.mkdir(parents=True, exist_ok=True)


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
            out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, generator=seed)
        else:
            out = pipe(prompt, num_inference_steps=30, generator=seed)
        
        heat_map = tc.compute_global_heat_map(factors=factors, head_idx=head_id, layer_idx=layer_id, time_idx=time_id)
        #print(heat_map.negative_heat_maps.shape)
        heat_map = tc.compute_global_heat_map(factors=factors, head_idx=head_id, layer_idx=layer_id, time_idx=time_id)
        #print(heat_map.negative_heat_maps.shape)

        plt.clf()
        plt.rcParams.update({'font.size': 16})
        fig, axs = plt.subplots(1, len(words) + 1, figsize=(5*(len(words)+1), 5+1))

        for ax in axs:
            ax.axis('off')
        
        axs[0].imshow(out.images[0])
        
        text = '' if layer_id is None else 'layer_id: ' + str(layer_id[0]) + ' ' + str(layer_id[-1])
        text += '' if factors is None else ' factors: ' + str(factors)
        text += '' if time_id is None else ' time_id: ' + str(time_id[0]) + ' ' + str(time_id[-1])
        text += '' if head_id is None else ' head_id: ' + str(head_id)
        
        axs[0].set_title(text)  # 调整这里的y值和字体

        for i, word in enumerate(words):
            heat_map_word = heat_map.compute_word_heat_map(word)
            heat_map_word.plot_overlay(out.images[0], ax=axs[i+1])

        plt.subplots_adjust(top=0.95)
        plt.savefig(f'{folder_path}/{title}.png', bbox_inches='tight')

print(f"png: {title}")
with open( f'{folder_path}/description.txt', 'a', encoding='utf-8') as file:
    file.write(f"png: {title}\n")
    file.write(f"prompt: {args.prompt}\n")             
    file.write(f"negative_prompt: {args.negative_prompt}\n")    
    file.write(f"words: {words}\n")          
    file.write(f"seed: {args.seed}\n\n")            
