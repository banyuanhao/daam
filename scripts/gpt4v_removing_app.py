from gpt4v_utils import gpt4_vision
from generate_tools import generate_image

import argparse

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--negative_prompt', type=str,default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--negative_start', type=int, default=6)
parser.add_argument('--outdir', type=str, default='.')
args = parser.parse_args()

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

def exact_remove(args):
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    negative_start = args.negative_start
    steps = args.steps
    seed = args.seed
    out_dir = args.outdir
    text_prompt = f'tell me \'yes\' if the {negative_prompt} is the image, otherwise, tell me \'no\'.'

    
    for negative_end in range(negative_start + 1,30):
        print(f'trying negative_start: {negative_start}, negative_end: {negative_end}')
        image = generate_image(prompt, negative_prompt, steps, seed, list(range(negative_start,negative_end+1)))
        
        context = gpt4_vision([image], text_prompt)
        
        if 'no' in context.lower():
            print('successfully removed')
            print(f'negative_start: {negative_start}, negative_end: {negative_end}')
            print(f'image saved to {out_dir}/{seed}_{negative_start}.png')
            image.save(f'{out_dir}/{seed}_{negative_start}_{negative_end}.png')
            return
    print('failed to remove')
    return

exact_remove(args)