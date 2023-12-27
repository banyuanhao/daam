import json
coco_classes_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic_light': 9, 'fire_hydrant': 10, 'stop_sign': 11, 'parking_meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports_ball': 32, 'kite': 33, 'baseball_bat': 34, 'baseball_glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis_racket': 38, 'bottle': 39, 'wine_glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot_dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted_plant': 58, 'bed': 59, 'dining_table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell_phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy_bear': 77, 'hair_dryer': 78, 'toothbrush': 79}
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

# open the json file and read the data
with open('dataset/coco/annotations/instances_val2014.json', 'r') as f:
    data = json.load(f)
    annotations_old = data['categories']
    
annotations = []
for dictionary in annotations_old:
    name = dictionary["name"]
    if name in coco_classes:
        tmp_dictionary = {
            "id": coco_classes_dict[name],
            "name": name,
            "supercategory": dictionary['supercategory']
        }
        annotations.append(tmp_dictionary)
print(annotations)
with open('dataset/ODFN/annotations/train.json', 'r') as f:
    old_data = json.load(f)
    old_data['categories'] = annotations
with open('dataset/ODFN/annotations/train.json', 'w') as f:
    # write the new data
    json.dump(old_data, f)