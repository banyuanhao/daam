import json
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer
import mmcv
import torch 
import os
import cv2
seeds_dict = {3502948: 0, 2414292: 1, 4013215: 2, 7661395: 3, 2728259: 4, 7977675: 5, 6926097: 6, 8223344: 7, 4338686: 8, 2630916: 9, 3548081: 10, 3710422: 11, 2285361: 12, 9638421: 13, 2837631: 14, 5468982: 15, 7955021: 16, 7197637: 17, 4206420: 18, 1347815: 19, 2957833: 20, 3326072: 21, 1813088: 22, 7965829: 23, 4708029: 24, 452169: 25, 1107126: 26, 8388604: 27, 9481161: 28, 8020003: 29, 2225075: 30, 1440263: 31, 29403: 32, 7099996: 33, 7851895: 34, 1106978: 35, 4053385: 36, 6882390: 37, 3322966: 38, 3668830: 39, 8613167: 40, 1315399: 41, 3121499: 42, 900759: 43, 7739336: 44, 1464588: 45, 1144945: 46, 39451: 47, 3131354: 48, 6971254: 49, 1088493: 50, 1700896: 51, 3760774: 52, 3410488: 53, 3129936: 54, 5309498: 55, 3698823: 56, 5970284: 57, 2569054: 58, 8264031: 59, 8663422: 60, 5174978: 61, 4041203: 62, 1690212: 63, 7695658: 64, 4857840: 65, 4395970: 66, 2970532: 67, 1313178: 68, 7409679: 69, 1242182: 70, 6902329: 71, 4582656: 72, 4123976: 73, 8158709: 74, 3033046: 75, 1634920: 76, 6750562: 77, 6337306: 78, 8317766: 79, 1618731: 80, 1518909: 81, 4798495: 82, 2620399: 83, 2423703: 84, 7285262: 85, 180696: 86, 8432894: 87, 3157912: 88, 7890161: 89, 5509442: 90, 6216034: 91, 7431925: 92, 7774348: 93, 6443781: 94, 6142998: 95, 3686770: 96, 8916284: 97, 9406101: 98, 7637527: 99}
seeds = [3502948, 2414292, 4013215, 7661395, 2728259, 7977675, 6926097, 8223344, 4338686, 2630916, 3548081, 3710422, 2285361, 9638421, 2837631, 5468982, 7955021, 7197637, 4206420, 1347815, 2957833, 3326072, 1813088, 7965829, 4708029, 452169, 1107126, 8388604, 9481161, 8020003, 2225075, 1440263, 29403, 7099996, 7851895, 1106978, 4053385, 6882390, 3322966, 3668830, 8613167, 1315399, 3121499, 900759, 7739336, 1464588, 1144945, 39451, 3131354, 6971254, 1088493, 1700896, 3760774, 3410488, 3129936, 5309498, 3698823, 5970284, 2569054, 8264031, 8663422, 5174978, 4041203, 1690212, 7695658, 4857840, 4395970, 2970532, 1313178, 7409679, 1242182, 6902329, 4582656, 4123976, 8158709, 3033046, 1634920, 6750562, 6337306, 8317766, 1618731, 1518909, 4798495, 2620399, 2423703, 7285262, 180696, 8432894, 3157912, 7890161, 5509442, 6216034, 7431925, 7774348, 6443781, 6142998, 3686770, 8916284, 9406101, 7637527] 
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
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_dryer', 'toothbrush'
    ]
spilt = 'val'
if spilt == 'test':
    seeds_sub = seeds[90:95]
elif spilt == 'val':
    seeds_sub = seeds[80:85]
elif spilt == 'train':
    seeds_sub = seeds[:40]
else:
    raise ValueError('category error')
    
with open(f'dataset/ODFN/{spilt}/annotations/{spilt}.json', 'r') as f:
    data = json.load(f)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    
def extract_ground(image_id):
    class_id = image_id //100000
    seed_id = image_id // 100 % 1000
    prompt_id = image_id % 100
    return class_id, seed_id, prompt_id


annotations_for_1_category = []
for annotation in annotations:
    image_id = annotation['image_id']
    category_id_truth, seed_id, prompt_id = extract_ground(image_id)
    score = annotation['score']
    category_id = annotation['category_id']
    bbox = annotation['bbox']
    id = annotation['id']
    rank = annotation['rank']
    if  category_id == category_id_truth and score > 0.8 and rank == 1:
        dict_tmp= {}
        dict_tmp['image_id'] = seed_id
        dict_tmp['category_id'] = 0
        dict_tmp['bbox'] = [i/8 for i in bbox]
        dict_tmp['score'] = score
        dict_tmp['rank'] = rank
        dict_tmp['id'] = id
        dict_tmp['iscrowd'] = 0
        dict_tmp['area'] = annotation['area'] / 64
        annotations_for_1_category.append(dict_tmp)
data['annotations'] = annotations_for_1_category

images_for_1_category = []
for seed in seeds_sub:
    image_tmp = {}
    image_tmp['id'] = seeds_dict[seed]
    image_tmp['file_name'] = f'{spilt}/noises/' + str(seed) + '.npy'
    image_tmp['width'] = 512 / 8
    image_tmp['height'] = 512 / 8
    images_for_1_category.append(image_tmp)
data['images'] = images_for_1_category

data['categories'] = [{'id': 0, 'name': 'object', 'supercategory': 'object'}]

# save annotations_for_80_category
with open(f'dataset/ODFN/{spilt}/annotations/{spilt}_for_1_category.json', 'w') as f:
    json.dump(data, f)

    
# cateset = set()
# for annotation in annotations:
#     cateset.add(annotation['category_id'])
    
# print(cateset)
# print(len(cateset))
    
# raise ValueError('stop')
# count_up = 0
# count_down = 0
# for annotation in annotations:

#     category_id = annotation['category_id']
#     bbox = annotation['bbox']
#     score = annotation['score']
#     rank = annotation['rank']
    
#     if rank == 1:
#         for image in images:
#             if image['id'] == annotation['image_id']:
#                 img_path = 'dataset/ODFN/' + image['file_name']
#                 print(img_path)
#                 break
    

# image = mmcv.imread(img_path)

# visualizer = Visualizer(image=image,save_dir='pics')

# visualizer.draw_bboxes(torch.tensor(bbox))
# visualizer.draw_texts(coco_classes[category_id],torch.tensor(bbox)[0:2])
# a = visualizer.get_image()
# cv2.imwrite('pics/a.png',a)
#visualizer.add_image('demo', a)