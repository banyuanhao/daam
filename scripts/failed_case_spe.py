from gpt4v_utils import gpt4_vision, gpt3_5_turbo, gpt4
from PIL import Image
from daam import trace, set_seed
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import json
seed = 0
verbose = True
save = True
image_save = False
prompt_name = 'add2'
time = [5,15]
model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

base_path = '/home/banyh2000/diffusion/daam/wrapupdata/remove/remove_data'

dict_ = []

    
def experoment_one(id, seed):
    caption = 'a living room'
    
    object = 'couch'
    time_id = [[0,3],[4,7],[8,11],[12,15],[16,19],[20,23],[24,27]]
    fig, axs = plt.subplots(2, len(time_id)+1, figsize=((len(time_id)+1)* 5,5*2))
    for row in axs:
        for ax in row:
            ax.axis('off')
    fig.tight_layout()
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(caption, num_inference_steps=30, generator=set_seed(seed))
            axs[0][0].imshow(out.images[0])
            for i, time_i in enumerate(time_id):
                heat_map = tc.compute_global_heat_map(time_idx=time_i)
                heat_map_word = heat_map.compute_word_heat_map('room')
                heat_map_word.plot_overlay(out.images[0], ax=axs[0][i+1])
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            out = pipe(caption, negative_prompt=object, num_inference_steps=30, generator=set_seed(seed))
            axs[1][0].imshow(out.images[0])
            for i, time_i in enumerate(time_id):
                heat_map = tc.compute_global_heat_map(time_idx=time_i)
                heat_map_word = heat_map.compute_word_heat_map('n:'+object)
                # convert 2D tensor heat_map_word.heatmap to PIL.Image
                heat_map_word.plot_overlay(out.images[0], ax=axs[1][i+1])
    

        fig.savefig(f'{base_path}/images/spe/{object}/remove_{id}_{seed}.png')
        dict_.append({'id':id, 'caption':caption,'seed':seed, 'object':object})
        with open(f'{base_path}/images/spe/{object}/failed.json', 'w') as f:
            json.dump(dict_, f)
        
    # clear matplotlib plots
    plt.close(fig)

np.random.seed(seed)
for i in range(80):
    seed_for_image = np.random.randint(0, 1000000)
    experoment_one(i, seed_for_image)
        