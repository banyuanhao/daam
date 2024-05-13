# mean activation value of the feature maps

import argparse
from daam import trace, set_seed
from diffusers import UnCLIPPipeline
# from models.diffuserpipeline import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import wandb
#
parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--prompt', type=str)
parser.add_argument('--negative_prompt', type=str, default='')
parser.add_argument('--seed', type=int,nargs='+', default=[0])
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--group', type=str)
args = parser.parse_args()

import torch
from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

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
prompt = "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape unbelievable; wave; wave shape spectacular"

with torch.no_grad():
    with trace(pipe) as tc:
        image = pipe(prompt=prompt,negative_prompt='water').images[0]
        image.save("wave.png")
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map('n:grass')
        heat_map.plot_overlay(image.images[0],ax = plt.gca())
        heat_ = tc.return_heat_map()

# prompt = args.prompt
# negative_prompt = args.negative_prompt
# seeds = args.seed
# steps = args.steps
# tags = args.tags



# if args.wandb:
#     wandb.login()
#     wandb.config = {"prompt": prompt,
#                     "negative prompt": negative_prompt, 
#                     "seeds": seeds,
#                     "steps": steps,
#                     "tags": tags,
#                     }
#     run = wandb.init(
#         project=args.project,
#         notes=args.note,
#         group=args.group,
#         config=wandb.config,
#     )

# ratio_list = []

# for seed in iter(seeds):
#     with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
#         with trace(pipe) as tc:
            
#             out = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps, generator=set_seed(seed),output_type='latent')
            
#             pos, neg = tc.compute_activation_ratio()
#             ratio = [neg[i]/pos[i] for i in range(len(pos))]
            
#             ratio_list.append(ratio)

# # with open('wrapupdata/ratio_adj.json','r') as f:
# #     data = json.load(f)
    
# # data.append({'prompt':prompt, 'negative_prompt':negative_prompt, 'seeds':seeds, 'steps':steps, 'ratio':ratio_list})

# # with open('wrapupdata/ratio_adj.json', 'w') as f:
# #     json.dump(data, f)