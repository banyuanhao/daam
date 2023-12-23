from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image,  StableDiffusionXLImg2ImgPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
import os

seeds = [3502948, 2414292, 4013215, 7661395, 2728259, 7977675, 6926097, 8223344, 4338686, 2630916, 3548081, 3710422, 2285361, 9638421, 2837631, 5468982, 7955021, 7197637, 4206420, 1347815, 2957833, 3326072, 1813088, 7965829, 4708029, 452169, 1107126, 8388604, 9481161, 8020003, 2225075, 1440263, 29403, 7099996, 7851895, 1106978, 4053385, 6882390, 3322966, 3668830, 8613167, 1315399, 3121499, 900759, 7739336, 1464588, 1144945, 39451, 3131354, 6971254, 1088493, 1700896, 3760774, 3410488, 3129936, 5309498, 3698823, 5970284, 2569054, 8264031, 8663422, 5174978, 4041203, 1690212, 7695658, 4857840, 4395970, 2970532, 1313178, 7409679, 1242182, 6902329, 4582656, 4123976, 8158709, 3033046, 1634920, 6750562, 6337306, 8317766, 1618731, 1518909, 4798495, 2620399, 2423703, 7285262, 180696, 8432894, 3157912, 7890161, 5509442, 6216034, 7431925, 7774348, 6443781, 6142998, 3686770, 8916284, 9406101, 7637527]

seeds = seeds[25:30]

def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen

model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'

# pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)  

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")


dataset_path = Path('dataset/ODFN')

prompts_path = dataset_path/'prompts'
prompts_names = os.listdir(prompts_path)

image_path = dataset_path/'images'

with torch.no_grad():
    for prompt_name in prompts_names:
        prompt_path = prompts_path/prompt_name
        image_class_path = image_path/prompt_name[0:-4]
        
        if not os.path.exists(image_class_path):
            os.makedirs(image_class_path)
            
        for seed in seeds:
            image_seed_path = image_class_path/str(seed)
            if not os.path.exists(image_seed_path):
                os.makedirs(image_seed_path)
            
            with open(prompt_path, 'r') as f:
                prompts = f.read()
                prompts = prompts.split('\n')
                name = prompts[0]
                prompts = prompts[1:]
                
            for k, prompt in enumerate(prompts):
                out = pipe(prompt=prompt, generator=set_seed(seed))
                out.images[0].save(image_seed_path/f'{name}_{seed}_{k}.png')

