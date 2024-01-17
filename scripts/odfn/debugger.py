from scipy.stats import normaltest, kstest
import numpy as np
import json
import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import TypeVar
from tqdm import tqdm
from scipy.stats import norm
import mmdet.datasets.coco
from mmdet.datasets.transforms import PackDetInputs
from mmcv.transforms import RandomResize,RandomFlip


with open('dataset/ODFN/version_1/train/annotations/train_for_1_category.json', 'r') as f:
    data = json.load(f)
    annotations = data['annotations']
    categories = data['categories']
    images = data['images']

print(images[0])
    
# count_up = np.zeros(80)
# count_down = np.zeros(80)
# diff = np.zeros(80)
# original = np.zeros(80)
    
# for i, annotation in tqdm(enumerate(annotations)):
#     image_id = annotation['image_id']
#     category_id_truth, seed_id, prompt_id = extract_ground(image_id)
#     score = annotation['score']
#     category_id = annotation['category_id']
#     bbox = annotation['bbox']
#     rank = annotation['rank']
#     id = annotation['id']
#     latent = randn_tensor((1, 4, 64, 64), generator=set_seed(seed_id))
#     latent_cond = randn_tensor((1, 4, 64, 64), generator=set_seed(0))
#     #latent = randn_tensor((1, 4, 64, 64), generator=set_seed(1))
#     bbox = [int(value)//8 for value in bbox]
    
#     if category_id_truth == category_id and score > 0.7 and rank == 1:
#         # get the latent in the bbox
#         latent_in = latent[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         # latent_in = latent[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
#         # apply normal test to the latent_in
#         latent_in = latent_in.flatten()
#         stat, p = normaltest(latent_in.cpu().numpy())
        
#         latent_in_cond = latent_cond[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         # apply normal test to the latent_in
#         latent_in_cond = latent_in_cond.flatten()
#         stat_cond, p_cond = kstest(latent_in_cond.cpu().numpy(),'norm')
#         original[category_id] += p

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
#         diff[category_id] += p_cond - p
#         count_down[category_id] += 1
#         if p_cond > p:
#             count_up[category_id] += 1
            
# ratio = [count_up[i]/count_down[i] for i in range(80)]
# print(ratio)
# print(diff)
# print(count_down)
# print([diff[i]/count_down[i] for i in range(80)])
# print([original[i]/count_down[i] for i in range(80)])
            
        
        
        
# # get a random np array
# img = np.random.randint(0,255,size = (64, 64))
# # save img as a np array
# np.save('pics/test.npy', img)
# # save img as png using mmcv
# # mmcv.imwrite(img, 'pics/test1.png')

# # instance_ = {'img_prefix': '/content/drive/MyDrive/test_data',
# #             'img_path': ['/home/banyh2000/diffusion/daam/daam/dataset/ODFN/train/images/apple/29403/apple_29403_3.png', '/home/banyh2000/diffusion/daam/daam/dataset/ODFN/train/images/apple/29403/apple_29403_2.png']}

# instance_ = {'img_prefix': '/content/drive/MyDrive/     test_data','img_path': '/home/banyh2000/diffusion/daam/daam/pics/test.npy'}
# transform = LoadImageFromNPY()
# instance_ = transform(instance_)
# print(type(instance_['img']))

# instance_ = {'img_prefix': '/content/drive/MyDrive/     test_data','img_path': ['/home/banyh2000/diffusion/daam/daam/pics/test1.png','/home/banyh2000/diffusion/daam/daam/pics/test1.png']}
# transform = LoadMultiChannelImageFromFiles()
# instance_ = transform(instance_)
# print(type(instance_['img']))

# import json
# with open('dataset/ODFN/test/annotations/test.json', 'r') as f:
#     data = json.load(f)
#     images = data['images']
#     annotations = data['annotations']
#     categories = data['categories']
    
# print(categories[0])