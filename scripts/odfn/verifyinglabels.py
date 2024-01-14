import json


from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer
import mmcv
import torch 
import os
import cv2
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
    
with open('dataset/ODFN/test/annotations/test.json', 'r') as f:
    data = json.load(f)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    
print(images[0])
raise ValueError('stop')
    
def extract_ground(image_id):
    class_id = image_id //100000
    seed_id = image_id // 100 % 1000
    prompt_id = image_id % 100
    return class_id, seed_id, prompt_id


annotations_for_80_category = []
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
        dict_tmp['category_id'] = category_id
        dict_tmp['bbox'] = bbox
        dict_tmp['score'] = score
        dict_tmp['rank'] = rank
        dict_tmp['id'] = id
        dict_tmp['area'] = annotation['area']
        annotations_for_80_category.append(dict_tmp)
data['annotations'] = annotations_for_80_category

images_for_80_category = []
for image in images:
    image_id = image['id']
    category_id_truth, seed_id, prompt_id = extract_ground(image_id)
    image_tmp = {}
    image_tmp['id'] = seed_id
    
# save annotations_for_80_category
with open('dataset/ODFN/test/annotations/test_for_80_category.json', 'w') as f:
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