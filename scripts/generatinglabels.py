from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer
import mmcv
import torch 
import os
import cv2
from pathlib import Path
from tqdm import tqdm

seeds = [3502948, 2414292, 4013215, 7661395, 2728259, 7977675, 6926097, 8223344, 4338686, 2630916, 3548081, 3710422, 2285361, 9638421, 2837631, 5468982, 7955021, 7197637, 4206420, 1347815, 2957833, 3326072, 1813088, 7965829, 4708029, 452169, 1107126, 8388604, 9481161, 8020003, 2225075, 1440263, 29403, 7099996, 7851895, 1106978, 4053385, 6882390, 3322966, 3668830, 8613167, 1315399, 3121499, 900759, 7739336, 1464588, 1144945, 39451, 3131354, 6971254, 1088493, 1700896, 3760774, 3410488, 3129936, 5309498, 3698823, 5970284, 2569054, 8264031, 8663422, 5174978, 4041203, 1690212, 7695658, 4857840, 4395970, 2970532, 1313178, 7409679, 1242182, 6902329, 4582656, 4123976, 8158709, 3033046, 1634920, 6750562, 6337306, 8317766, 1618731, 1518909, 4798495, 2620399, 2423703, 7285262, 180696, 8432894, 3157912, 7890161, 5509442, 6216034, 7431925, 7774348, 6443781, 6142998, 3686770, 8916284, 9406101, 7637527]

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
annotations = []
images = []
config_file = 'modelpara/det/gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco.py'
checkpoint_file = 'modelpara/det/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'
model = init_detector(config_file, checkpoint_file, device='cuda')

dataset_path = Path('dataset/ODFN')
image_path = dataset_path/'images'
class_names = os.listdir(image_path)
for class_name in tqdm(class_names):
    class_path = image_path/class_name
    seeds = os.listdir(class_path)
    for seed in seeds:
        seed_path = class_path/seed
        image_ins_names = os.listdir(seed_path)
        for _, image_ins_name in enumerate(image_ins_names):
            img_path = seed_path/image_ins_name
            result = inference_detector(model, img_path)
            j = int(image_ins_name[0])
            image_id = str(coco_classes_dict[class_name]).zfill(2)+str(seeds_dict[int(seed)]).zfill(3)+str(j).zfill(2)
            
            image = {
                'id': int(image_id),
                'width': 512,
                'height': 512,
                'file_name': str(img_path),
            }
            images.append(image)
            
            for p, score in enumerate(result.pred_instances.scores):
                if p > 5: 
                    break
                label = int(result.pred_instances.labels[p])
                label_name = coco_classes[label]
                bbox = result.pred_instances.bboxes[p]
                annotation = {
                    'image_id': int(image_id),
                    'score': score.item(),
                    'category_id': label,
                    'bbox': bbox.cpu().numpy().tolist(),
                    'id': int(image_id)*10+p,
                    'rank': p+1,
                }
                annotations.append(annotation)
                
    save_dict = {
        'images': images,
        'annotations': annotations,
    }

    # save save_dict to json file
    import json
    if not os.path.exists('dataset/ODFN/annotations'):
        os.makedirs('dataset/ODFN/annotations')
    with open('dataset/ODFN/annotations/instances_val2017.json', 'w') as f:
        json.dump(save_dict, f)