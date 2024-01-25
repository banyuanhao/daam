# test the variance of the labels
import json
import torch
import mmcv
from mmengine.visualization import Visualizer
import math
import cv2
from pathlib import Path
super_class = ['outdoor', 'indoor', 'vehicle', 'person', 'electronic', 'animal', 'food', 'appliance', 'furniture', 'accessory', 'kitchen', 'sports']
super_dict = {'outdoor': 0, 'indoor': 1, 'vehicle': 2, 'person': 3, 'electronic': 4, 'animal': 5, 'food': 6, 'appliance': 7, 'furniture': 8, 'accessory': 9, 'kitchen': 10, 'sports': 11}
# vehicle, electrinic, animal, furniture
category_id_to_superclass_id = {0: 3, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5, 23: 5, 24: 9, 25: 9, 26: 9, 27: 9, 28: 9, 29: 11, 30: 11, 31: 11, 32: 11, 33: 11, 34: 11, 35: 11, 36: 11, 37: 11, 38: 11, 39: 10, 40: 10, 41: 10, 42: 10, 43: 10, 44: 10, 45: 10, 46: 6, 47: 6, 48: 6, 49: 6, 50: 6, 51: 6, 52: 6, 53: 6, 54: 6, 55: 6, 56: 8, 57: 8, 58: 8, 59: 8, 60: 8, 61: 8, 62: 4, 63: 4, 64: 4, 65: 4, 66: 4, 67: 4, 68: 7, 69: 7, 70: 7, 71: 7, 72: 7, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 78: 1, 79: 1}

seeds = [3502948, 2414292, 4013215, 7661395, 2728259, 7977675, 6926097, 8223344, 4338686, 2630916, 3548081, 3710422, 2285361, 9638421, 2837631, 5468982, 7955021, 7197637, 4206420, 1347815, 2957833, 3326072, 1813088, 7965829, 4708029, 452169, 1107126, 8388604, 9481161, 8020003, 2225075, 1440263, 29403, 7099996, 7851895, 1106978, 4053385, 6882390, 3322966, 3668830, 8613167, 1315399, 3121499, 900759, 7739336, 1464588, 1144945, 39451, 3131354, 6971254, 1088493, 1700896, 3760774, 3410488, 3129936, 5309498, 3698823, 5970284, 2569054, 8264031, 8663422, 5174978, 4041203, 1690212, 7695658, 4857840, 4395970, 2970532, 1313178, 7409679, 1242182, 6902329, 4582656, 4123976, 8158709, 3033046, 1634920, 6750562, 6337306, 8317766, 1618731, 1518909, 4798495, 2620399, 2423703, 7285262, 180696, 8432894, 3157912, 7890161, 5509442, 6216034, 7431925, 7774348, 6443781, 6142998, 3686770, 8916284, 9406101, 7637527]
seeds = seeds[:40]

seeds_dict = {3502948: 0, 2414292: 1, 4013215: 2, 7661395: 3, 2728259: 4, 7977675: 5, 6926097: 6, 8223344: 7, 4338686: 8, 2630916: 9, 3548081: 10, 3710422: 11, 2285361: 12, 9638421: 13, 2837631: 14, 5468982: 15, 7955021: 16, 7197637: 17, 4206420: 18, 1347815: 19, 2957833: 20, 3326072: 21, 1813088: 22, 7965829: 23, 4708029: 24, 452169: 25, 1107126: 26, 8388604: 27, 9481161: 28, 8020003: 29, 2225075: 30, 1440263: 31, 29403: 32, 7099996: 33, 7851895: 34, 1106978: 35, 4053385: 36, 6882390: 37, 3322966: 38, 3668830: 39, 8613167: 40, 1315399: 41, 3121499: 42, 900759: 43, 7739336: 44, 1464588: 45, 1144945: 46, 39451: 47, 3131354: 48, 6971254: 49, 1088493: 50, 1700896: 51, 3760774: 52, 3410488: 53, 3129936: 54, 5309498: 55, 3698823: 56, 5970284: 57, 2569054: 58, 8264031: 59, 8663422: 60, 5174978: 61, 4041203: 62, 1690212: 63, 7695658: 64, 4857840: 65, 4395970: 66, 2970532: 67, 1313178: 68, 7409679: 69, 1242182: 70, 6902329: 71, 4582656: 72, 4123976: 73, 8158709: 74, 3033046: 75, 1634920: 76, 6750562: 77, 6337306: 78, 8317766: 79, 1618731: 80, 1518909: 81, 4798495: 82, 2620399: 83, 2423703: 84, 7285262: 85, 180696: 86, 8432894: 87, 3157912: 88, 7890161: 89, 5509442: 90, 6216034: 91, 7431925: 92, 7774348: 93, 6443781: 94, 6142998: 95, 3686770: 96, 8916284: 97, 9406101: 98, 7637527: 99}

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

coco_classes_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic_light': 9, 'fire_hydrant': 10, 'stop_sign': 11, 'parking_meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports_ball': 32, 'kite': 33, 'baseball_bat': 34, 'baseball_glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis_racket': 38, 'bottle': 39, 'wine_glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot_dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted_plant': 58, 'bed': 59, 'dining_table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell_phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy_bear': 77, 'hair_dryer': 78, 'toothbrush': 79}

# open the json file and read the data
with open('dataset/ODFN/train/annotations/train.json', 'r') as f:
    data = json.load(f)
    
annotations = data['annotations']
images = data['images']


# selecting
## the top 1 bounding box matching the ground truth scores > 0.5
## the top 1 bounding boxes matching the ground truth

# statistics:
## constant seed_id                         totally 40  
## constant seed_id & catergory_id_truth    totally 40*10
## constant category_id                     totally 80
## constant category_id_truth               totall  80
## constant category_id & prompt_id 80      totally 80*10
## total                                    totally 80*40*10

def get_dict(statics_mode):
    if statics_mode == 'seed_id':
        dictionary = {key: [] for key in seeds}
    elif statics_mode == 'category_id_truth':
        dictionary = {key: [] for key in range(80)}
    elif statics_mode == 'seed_id_superclass':
        dictionary = {(key1, key2): [] for key1 in seeds for key2 in range(len(super_class))}
    elif statics_mode == 'category_id_prompt':
        dictionary = {(key1, key2): [] for key1 in range(80) for key2 in range(10)}
    else:
        raise ValueError('statics_mode is not defined')
    return dictionary

def sorting_box(dictionary, category_id_truth, seed_id, prompt_id, category_id, bbox, statics_mode):
    if statics_mode == 'seed_id':
        dictionary[seed_id].append(bbox)
    elif statics_mode == 'category_id_truth':
        dictionary[category_id_truth].append(bbox)
    elif statics_mode == 'seed_id_superclass':
        dictionary[(seed_id, category_id_to_superclass_id[category_id_truth])].append(bbox)
    elif statics_mode == 'category_id_prompt':
        dictionary[(category_id_truth, prompt_id)].append(bbox)
    else:
        raise ValueError('statics_mode is not defined')
    return dictionary

def extract_ground(image_id):
    class_id = image_id //100000
    seed_id = image_id // 100 % 1000
    prompt_id = image_id % 100
    return class_id, seed_id, prompt_id

#dataset_path = Path('dataset/ODFN')
select_mode = 'top1'
statics_mode = 'category_id_prompt'
#statics_mode = 'seed_id'
count = 0
dictionary = get_dict(statics_mode)
for i, annotation in enumerate(annotations):
    image_id = annotation['image_id']
    category_id_truth, seed_id, prompt_id = extract_ground(image_id)
    score = annotation['score']
    category_id = annotation['category_id']
    bbox = annotation['bbox']
    id = annotation['id']
    rank = annotation['rank']
    if  category_id == category_id_truth and score > 0.6 and rank == 1:
        count += 1
        dictionary = sorting_box(dictionary, category_id_truth, seeds[seed_id], prompt_id, category_id, bbox, statics_mode)
        
statics = {}
countx = 0
county = 0
for key, value in dictionary.items():
    if len(value) == 0:
        continue
    tensor = torch.tensor(value)
    tensorx = tensor[:,[0,2]].mean(dim=1).var().item()
    tensory = tensor[:,[1,3]].mean(dim=1).var().item()
    statics[key] = [tensorx, tensory]
    if tensorx > 6200:
        countx += 1
    if tensory > 3300:
        county += 1
        

# key_value_pairs = {key: value[0] for key, value in statics.items()}

# # 根据值排序键
# sorted_keys = sorted(key_value_pairs, key=lambda x: key_value_pairs[x])

# # 现在 sorted_keys 是一个列表，其中包含按值排序的键
# print(sorted_keys)
# for i in range(80):
#     if category_id_to_superclass_id[i] == 6:
#         print(statics[i])
print(statics)


mean = [value  for key, value in statics.items() if not math.isnan(value[0]) and not math.isnan(value[1])]
mean = torch.mean(torch.tensor(mean), dim=0)
print(mean)
# print(statics)
    
# print(countx)
# print(county)

# tensor([5706.6182, 2847.8071])

# print(dictionary)
# print(count)
# img_path = dataset_path/'images'/coco_classes[category_id_truth]/str(seeds[seed_id])/(coco_classes[category_id_truth]+'_'+str(seeds[seed_id])+'_'+str(prompt_id)+'.png')
# image = mmcv.imread(img_path, channel_order='rgb')
# visualizer = Visualizer(image=image,save_dir='pics')
# print(score)
# visualizer.draw_bboxes(torch.tensor(bbox))
# visualizer.draw_texts(coco_classes[category_id],torch.tensor(bbox[0:2]))
# a = visualizer.get_image()
# cv2.imwrite('pics/a.png',a)