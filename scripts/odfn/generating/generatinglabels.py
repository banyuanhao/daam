from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer
import mmcv
import torch 
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from utils_odfn import coco_classes,seeds, coco_classes_dict
from utils_odfn import categories as categories_old

    

categories = []
for dictionary in categories_old:
    name = dictionary["name"]
    if name in coco_classes:
        tmp_dictionary = {
            "id": coco_classes_dict[name],
            "name": name,
            "supercategory": dictionary['supercategory']
        }
        categories.append(tmp_dictionary)
        
annotations = []
images = []
config_file = 'modelpara/det/gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco.py'
checkpoint_file = 'modelpara/det/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'
model = init_detector(config_file, checkpoint_file, device='cuda')

spilt = 'test'

dataset_path = Path(f'dataset/ODFN/{spilt}')
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
            j = int(image_ins_name[-5])
            image_id = str(coco_classes_dict[class_name]).zfill(2)+str(seeds_dict[int(seed)]).zfill(3)+str(j).zfill(2)
            
            image = {
                'id': int(image_id),
                'width': 512,
                'height': 512,
                'file_name': str(Path(os.path.join(*img_path.parts[2:]))),
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
                    'area': (bbox[2].item()-bbox[0].item())*(bbox[3].item()-bbox[1].item()),
                }
                annotations.append(annotation)
                
    save_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    # save save_dict to json file
    if not os.path.exists(f'dataset/ODFN/{spilt}/annotations'):
        os.makedirs(f'dataset/ODFN/{spilt}/annotations')
    with open(f'dataset/ODFN/{spilt}/annotations/{spilt}.json', 'w') as f:
        json.dump(save_dict, f)