import numpy as np
from typing import TypeVar
T = TypeVar('T')
import torch
import random
import pickle
import math
import matplotlib.pyplot as plt
seeds_plus = np.load('scripts/odfn/generating/seeds.npy').tolist()

seeds_plus_dict = {}
for i, seed in enumerate(seeds_plus):
    seeds_plus_dict[seed] = i
seeds_plus_shuffled = np.load('scripts/odfn/generating/seeds_plus_shuffle.npy').tolist()
seeds_plus_shuffled_dict = {}
for i in range(len(seeds_plus)):
    seeds_plus_shuffled_dict[seeds_plus_shuffled[i]] = seeds_plus[i]

    
# print(seeds_plus_dict)
# raise ValueError('stop')
# with open('scripts/odfn/generating/seeds_dict.pkl', 'wb') as f:
#     pickle.dump(seeds_plus, f)
# #seeds_plus_dict = np.load('scripts/odfn/generating/seeds_dict.npy', allow_pickle=True)
# with open('scripts/odfn/generating/seeds_dict.pkl', 'rb') as f:
#     seeds_plus_dict = pickle.load(f)

coco_classes_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic_light': 9, 'fire_hydrant': 10, 'stop_sign': 11, 'parking_meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports_ball': 32, 'kite': 33, 'baseball_bat': 34, 'baseball_glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis_racket': 38, 'bottle': 39, 'wine_glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot_dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted_plant': 58, 'bed': 59, 'dining_table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell_phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy_bear': 77, 'hair_dryer': 78, 'toothbrush': 79}
seeds = [3502948, 2414292, 4013215, 7661395, 2728259, 7977675, 6926097, 8223344, 4338686, 2630916, 3548081, 3710422, 2285361, 9638421, 2837631, 5468982, 7955021, 7197637, 4206420, 1347815, 2957833, 3326072, 1813088, 7965829, 4708029, 452169, 1107126, 8388604, 9481161, 8020003, 2225075, 1440263, 29403, 7099996, 7851895, 1106978, 4053385, 6882390, 3322966, 3668830, 8613167, 1315399, 3121499, 900759, 7739336, 1464588, 1144945, 39451, 3131354, 6971254, 1088493, 1700896, 3760774, 3410488, 3129936, 5309498, 3698823, 5970284, 2569054, 8264031, 8663422, 5174978, 4041203, 1690212, 7695658, 4857840, 4395970, 2970532, 1313178, 7409679, 1242182, 6902329, 4582656, 4123976, 8158709, 3033046, 1634920, 6750562, 6337306, 8317766, 1618731, 1518909, 4798495, 2620399, 2423703, 7285262, 180696, 8432894, 3157912, 7890161, 5509442, 6216034, 7431925, 7774348, 6443781, 6142998, 3686770, 8916284, 9406101, 7637527]
seeds_dict = {3502948: 0, 2414292: 1, 4013215: 2, 7661395: 3, 2728259: 4, 7977675: 5, 6926097: 6, 8223344: 7, 4338686: 8, 2630916: 9, 3548081: 10, 3710422: 11, 2285361: 12, 9638421: 13, 2837631: 14, 5468982: 15, 7955021: 16, 7197637: 17, 4206420: 18, 1347815: 19, 2957833: 20, 3326072: 21, 1813088: 22, 7965829: 23, 4708029: 24, 452169: 25, 1107126: 26, 8388604: 27, 9481161: 28, 8020003: 29, 2225075: 30, 1440263: 31, 29403: 32, 7099996: 33, 7851895: 34, 1106978: 35, 4053385: 36, 6882390: 37, 3322966: 38, 3668830: 39, 8613167: 40, 1315399: 41, 3121499: 42, 900759: 43, 7739336: 44, 1464588: 45, 1144945: 46, 39451: 47, 3131354: 48, 6971254: 49, 1088493: 50, 1700896: 51, 3760774: 52, 3410488: 53, 3129936: 54, 5309498: 55, 3698823: 56, 5970284: 57, 2569054: 58, 8264031: 59, 8663422: 60, 5174978: 61, 4041203: 62, 1690212: 63, 7695658: 64, 4857840: 65, 4395970: 66, 2970532: 67, 1313178: 68, 7409679: 69, 1242182: 70, 6902329: 71, 4582656: 72, 4123976: 73, 8158709: 74, 3033046: 75, 1634920: 76, 6750562: 77, 6337306: 78, 8317766: 79, 1618731: 80, 1518909: 81, 4798495: 82, 2620399: 83, 2423703: 84, 7285262: 85, 180696: 86, 8432894: 87, 3157912: 88, 7890161: 89, 5509442: 90, 6216034: 91, 7431925: 92, 7774348: 93, 6443781: 94, 6142998: 95, 3686770: 96, 8916284: 97, 9406101: 98, 7637527: 99}
categories_origin = [{'id': 0, 'name': 'person', 'supercategory': 'person'}, {'id': 1, 'name': 'bicycle', 'supercategory': 'vehicle'}, {'id': 2, 'name': 'car', 'supercategory': 'vehicle'}, {'id': 3, 'name': 'motorcycle', 'supercategory': 'vehicle'}, {'id': 4, 'name': 'airplane', 'supercategory': 'vehicle'}, {'id': 5, 'name': 'bus', 'supercategory': 'vehicle'}, {'id': 6, 'name': 'train', 'supercategory': 'vehicle'}, {'id': 7, 'name': 'truck', 'supercategory': 'vehicle'}, {'id': 8, 'name': 'boat', 'supercategory': 'vehicle'}, {'id': 9, 'name': 'traffic_light', 'supercategory': 'outdoor'}, {'id': 10, 'name': 'fire_hydrant', 'supercategory': 'outdoor'}, {'id': 11, 'name': 'stop_sign', 'supercategory': 'outdoor'}, {'id': 12, 'name': 'parking_meter', 'supercategory': 'outdoor'}, {'id': 13, 'name': 'bench', 'supercategory': 'outdoor'}, {'id': 14, 'name': 'bird', 'supercategory': 'animal'}, {'id': 15, 'name': 'cat', 'supercategory': 'animal'}, {'id': 16, 'name': 'dog', 'supercategory': 'animal'}, {'id': 17, 'name': 'horse', 'supercategory': 'animal'}, {'id': 18, 'name': 'sheep', 'supercategory': 'animal'}, {'id': 19, 'name': 'cow', 'supercategory': 'animal'}, {'id': 20, 'name': 'elephant', 'supercategory': 'animal'}, {'id': 21, 'name': 'bear', 'supercategory': 'animal'}, {'id': 22, 'name': 'zebra', 'supercategory': 'animal'}, {'id': 23, 'name': 'giraffe', 'supercategory': 'animal'}, {'id': 24, 'name': 'backpack', 'supercategory': 'accessory'}, {'id': 25, 'name': 'umbrella', 'supercategory': 'accessory'}, {'id': 26, 'name': 'handbag', 'supercategory': 'accessory'}, {'id': 27, 'name': 'tie', 'supercategory': 'accessory'}, {'id': 28, 'name': 'suitcase', 'supercategory': 'accessory'}, {'id': 29, 'name': 'frisbee', 'supercategory': 'sports'}, {'id': 30, 'name': 'skis', 'supercategory': 'sports'}, {'id': 31, 'name': 'snowboard', 'supercategory': 'sports'}, {'id': 32, 'name': 'sports_ball', 'supercategory': 'sports'}, {'id': 33, 'name': 'kite', 'supercategory': 'sports'}, {'id': 34, 'name': 'baseball_bat', 'supercategory': 'sports'}, {'id': 35, 'name': 'baseball_glove', 'supercategory': 'sports'}, {'id': 36, 'name': 'skateboard', 'supercategory': 'sports'}, {'id': 37, 'name': 'surfboard', 'supercategory': 'sports'}, {'id': 38, 'name': 'tennis_racket', 'supercategory': 'sports'}, {'id': 39, 'name': 'bottle', 'supercategory': 'kitchen'}, {'id': 40, 'name': 'wine_glass', 'supercategory': 'kitchen'}, {'id': 41, 'name': 'cup', 'supercategory': 'kitchen'}, {'id': 42, 'name': 'fork', 'supercategory': 'kitchen'}, {'id': 43, 'name': 'knife', 'supercategory': 'kitchen'}, {'id': 44, 'name': 'spoon', 'supercategory': 'kitchen'}, {'id': 45, 'name': 'bowl', 'supercategory': 'kitchen'}, {'id': 46, 'name': 'banana', 'supercategory': 'food'}, {'id': 47, 'name': 'apple', 'supercategory': 'food'}, {'id': 48, 'name': 'sandwich', 'supercategory': 'food'}, {'id': 49, 'name': 'orange', 'supercategory': 'food'}, {'id': 50, 'name': 'broccoli', 'supercategory': 'food'}, {'id': 51, 'name': 'carrot', 'supercategory': 'food'}, {'id': 52, 'name': 'hot_dog', 'supercategory': 'food'}, {'id': 53, 'name': 'pizza', 'supercategory': 'food'}, {'id': 54, 'name': 'donut', 'supercategory': 'food'}, {'id': 55, 'name': 'cake', 'supercategory': 'food'}, {'id': 56, 'name': 'chair', 'supercategory': 'furniture'}, {'id': 57, 'name': 'couch', 'supercategory': 'furniture'}, {'id': 58, 'name': 'potted_plant', 'supercategory': 'furniture'}, {'id': 59, 'name': 'bed', 'supercategory': 'furniture'}, {'id': 60, 'name': 'dining_table', 'supercategory': 'furniture'}, {'id': 61, 'name': 'toilet', 'supercategory': 'furniture'}, {'id': 62, 'name': 'tv', 'supercategory': 'electronic'}, {'id': 63, 'name': 'laptop', 'supercategory': 'electronic'}, {'id': 64, 'name': 'mouse', 'supercategory': 'electronic'}, {'id': 65, 'name': 'remote', 'supercategory': 'electronic'}, {'id': 66, 'name': 'keyboard', 'supercategory': 'electronic'}, {'id': 67, 'name': 'cell_phone', 'supercategory': 'electronic'}, {'id': 68, 'name': 'microwave', 'supercategory': 'appliance'}, {'id': 69, 'name': 'oven', 'supercategory': 'appliance'}, {'id': 70, 'name': 'toaster', 'supercategory': 'appliance'}, {'id': 71, 'name': 'sink', 'supercategory': 'appliance'}, {'id': 72, 'name': 'refrigerator', 'supercategory': 'appliance'}, {'id': 73, 'name': 'book', 'supercategory': 'indoor'}, {'id': 74, 'name': 'clock', 'supercategory': 'indoor'}, {'id': 75, 'name': 'vase', 'supercategory': 'indoor'}, {'id': 76, 'name': 'scissors', 'supercategory': 'indoor'}, {'id': 77, 'name': 'teddy_bear', 'supercategory': 'indoor'}, {'id': 78, 'name': 'hair_dryer', 'supercategory': 'indoor'}, {'id': 79, 'name': 'toothbrush', 'supercategory': 'indoor'}]
coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
        'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 
        'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 

        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_dryer', 'toothbrush'
    ]
version_2_classes = ['stop_sign', 'sports_ball', 'bear', 'handbag','baseball_glove']
version_1_classes = coco_classes
def seeds_spilt(spilt):
    if spilt == 'test':
        seeds_sub = seeds[90:95]
    elif spilt == 'val':
        seeds_sub = seeds[80:85]
    elif spilt == 'train':
        seeds_sub = seeds[:40]
    else:
        raise ValueError('category error')
    return seeds_sub

def seeds_plus_spilt(spilt):
    if spilt == 'test':
        seeds_sub = seeds_plus[18500:20000]
    elif spilt == 'val':
        seeds_sub = seeds_plus[17500:18500]
    elif spilt == 'train':
        seeds_sub = seeds_plus[:17500]
    else:
        raise ValueError('category error')
    return seeds_sub

def extract_ground_seeds(image_id):
    class_id = image_id //100000
    seed_id = image_id // 100 % 1000
    prompt_id = image_id % 100
    return class_id, seed_id, prompt_id

def extract_ground_seeds_plus(image_id):
    class_id = image_id //1000000
    seed_id = image_id // 10 % 100000
    prompt_id = image_id % 10
    return class_id, seed_id, prompt_id

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

def get_plt(num):
    plt.clf()
    plt.rcParams.update({'font.size': 12})
    if num < 5:
        fig, axs = plt.subplots(1, num , figsize=(5*num, 5+1))
        for ax in axs:
            ax.axis('off')
    else:
        fig, axs = plt.subplots(math.ceil(num / 4),4)

        for i in range(math.ceil(num / 4)):
            for j in range(4):
                ax = axs[i][j]
                ax.axis('off')
    plt.subplots_adjust(top=0.95)  
    return plt, fig, axs