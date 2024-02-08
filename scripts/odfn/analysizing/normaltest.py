from scipy.stats import normaltest
import numpy as np
import json
import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import TypeVar
import random
from tqdm import tqdm
T = TypeVar('T')

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

    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)

    return gen


def extract_ground(image_id):
    class_id = image_id //100000
    seed_id = image_id // 100 % 1000
    prompt_id = image_id % 100
    return class_id, seed_id, prompt_id

with open('dataset/ODFN/train/annotations/train.json', 'r') as f:
    data = json.load(f)
    annotations = data['annotations']
    categories = data['categories']
    images = data['images']
    
    
count_up = np.zeros(80)
count_down = np.zeros(80)
diff = np.zeros(80)
original = np.zeros(80)
    
for i, annotation in tqdm(enumerate(annotations)):
    image_id = annotation['image_id']
    category_id_truth, seed_id, prompt_id = extract_ground(image_id)
    score = annotation['score']
    category_id = annotation['category_id']
    bbox = annotation['bbox']
    rank = annotation['rank']
    id = annotation['id']
    latent = randn_tensor((1, 4, 64, 64), generator=set_seed(seed_id))
    latent_cond = randn_tensor((1, 4, 64, 64), generator=set_seed(0))
    #latent = randn_tensor((1, 4, 64, 64), generator=set_seed(1))
    bbox = [int(value)//8 for value in bbox]
    
    if category_id_truth == category_id and score > 0.7 and rank == 1:
        # get the latent in the bbox
        latent_in = latent[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # latent_in = latent[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
        # apply normal test to the latent_in
        latent_in = latent_in.flatten()
        stat, p = normaltest(latent_in.cpu().numpy())
        
        latent_in_cond = latent_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # apply normal test to the latent_in
        latent_in_cond = latent_in_cond.flatten()
        stat_cond, p_cond = normaltest(latent_in_cond.cpu().numpy())
        original[category_id] += p

        #print('image_id: ', image_id)
        #print('category_id_truth: ', category_id_truth)
        #print('seed_id: ', seed_id)
        #print('prompt_id: ', prompt_id)
        #print('score: ', score)
        #print('category_id: ', category_id)
        #print('bbox: ', bbox)
        #print('rank: ', rank)
        #print('id: ', id)
        #print('latent_in: ', latent_in)
        # print('stat: ', stat)
        # print('p: ', p)
        # print('stat_cond: ', stat_cond)
        # print('p_cond: ', p_cond)
        diff[category_id] += p_cond - p
        count_down[category_id] += 1
        if p_cond > p:
            count_up[category_id] += 1
            
ratio = [count_up[i]/count_down[i] for i in range(80)]
print(ratio)
print(diff)
print(count_down)
print([diff[i]/count_down[i] for i in range(80)])
print([original[i]/count_down[i] for i in range(80)])
            
        