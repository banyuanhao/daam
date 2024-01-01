import json
# coco_classes_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic_light': 9, 'fire_hydrant': 10, 'stop_sign': 11, 'parking_meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports_ball': 32, 'kite': 33, 'baseball_bat': 34, 'baseball_glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis_racket': 38, 'bottle': 39, 'wine_glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot_dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted_plant': 58, 'bed': 59, 'dining_table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell_phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy_bear': 77, 'hair_dryer': 78, 'toothbrush': 79}
# coco_classes = [
#         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#         'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
#         'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
#         'snowboard',
#         'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
#         'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
#         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#         'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
#         'chair', 'couch', 
#         'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
#         'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
#         'oven', 
        
#         'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#         'scissors', 'teddy_bear', 'hair_dryer', 'toothbrush'
#     ]

# # open the json file and read the data
# with open('dataset/coco/annotations/instances_val2014.json', 'r') as f:
#     data = json.load(f)
#     categories_old = data['categories']
    
# print(categories_old)
# annotations = []
# for dictionary in categories_old:
#     name = dictionary["name"].replace(' ', '_')
#     if name in coco_classes:
#         tmp_dictionary = {
#             "id": coco_classes_dict[name],
#             "name": name,
#             "supercategory": dictionary['supercategory']
#         }
#         annotations.append(tmp_dictionary)
#     else:
#         print(name)
        
# print(annotations)
        
# raise ValueError('stop')
categories = [{'id': 0, 'name': 'person', 'supercategory': 'person'}, {'id': 1, 'name': 'bicycle', 'supercategory': 'vehicle'}, {'id': 2, 'name': 'car', 'supercategory': 'vehicle'}, {'id': 3, 'name': 'motorcycle', 'supercategory': 'vehicle'}, {'id': 4, 'name': 'airplane', 'supercategory': 'vehicle'}, {'id': 5, 'name': 'bus', 'supercategory': 'vehicle'}, {'id': 6, 'name': 'train', 'supercategory': 'vehicle'}, {'id': 7, 'name': 'truck', 'supercategory': 'vehicle'}, {'id': 8, 'name': 'boat', 'supercategory': 'vehicle'}, {'id': 9, 'name': 'traffic_light', 'supercategory': 'outdoor'}, {'id': 10, 'name': 'fire_hydrant', 'supercategory': 'outdoor'}, {'id': 11, 'name': 'stop_sign', 'supercategory': 'outdoor'}, {'id': 12, 'name': 'parking_meter', 'supercategory': 'outdoor'}, {'id': 13, 'name': 'bench', 'supercategory': 'outdoor'}, {'id': 14, 'name': 'bird', 'supercategory': 'animal'}, {'id': 15, 'name': 'cat', 'supercategory': 'animal'}, {'id': 16, 'name': 'dog', 'supercategory': 'animal'}, {'id': 17, 'name': 'horse', 'supercategory': 'animal'}, {'id': 18, 'name': 'sheep', 'supercategory': 'animal'}, {'id': 19, 'name': 'cow', 'supercategory': 'animal'}, {'id': 20, 'name': 'elephant', 'supercategory': 'animal'}, {'id': 21, 'name': 'bear', 'supercategory': 'animal'}, {'id': 22, 'name': 'zebra', 'supercategory': 'animal'}, {'id': 23, 'name': 'giraffe', 'supercategory': 'animal'}, {'id': 24, 'name': 'backpack', 'supercategory': 'accessory'}, {'id': 25, 'name': 'umbrella', 'supercategory': 'accessory'}, {'id': 26, 'name': 'handbag', 'supercategory': 'accessory'}, {'id': 27, 'name': 'tie', 'supercategory': 'accessory'}, {'id': 28, 'name': 'suitcase', 'supercategory': 'accessory'}, {'id': 29, 'name': 'frisbee', 'supercategory': 'sports'}, {'id': 30, 'name': 'skis', 'supercategory': 'sports'}, {'id': 31, 'name': 'snowboard', 'supercategory': 'sports'}, {'id': 32, 'name': 'sports_ball', 'supercategory': 'sports'}, {'id': 33, 'name': 'kite', 'supercategory': 'sports'}, {'id': 34, 'name': 'baseball_bat', 'supercategory': 'sports'}, {'id': 35, 'name': 'baseball_glove', 'supercategory': 'sports'}, {'id': 36, 'name': 'skateboard', 'supercategory': 'sports'}, {'id': 37, 'name': 'surfboard', 'supercategory': 'sports'}, {'id': 38, 'name': 'tennis_racket', 'supercategory': 'sports'}, {'id': 39, 'name': 'bottle', 'supercategory': 'kitchen'}, {'id': 40, 'name': 'wine_glass', 'supercategory': 'kitchen'}, {'id': 41, 'name': 'cup', 'supercategory': 'kitchen'}, {'id': 42, 'name': 'fork', 'supercategory': 'kitchen'}, {'id': 43, 'name': 'knife', 'supercategory': 'kitchen'}, {'id': 44, 'name': 'spoon', 'supercategory': 'kitchen'}, {'id': 45, 'name': 'bowl', 'supercategory': 'kitchen'}, {'id': 46, 'name': 'banana', 'supercategory': 'food'}, {'id': 47, 'name': 'apple', 'supercategory': 'food'}, {'id': 48, 'name': 'sandwich', 'supercategory': 'food'}, {'id': 49, 'name': 'orange', 'supercategory': 'food'}, {'id': 50, 'name': 'broccoli', 'supercategory': 'food'}, {'id': 51, 'name': 'carrot', 'supercategory': 'food'}, {'id': 52, 'name': 'hot_dog', 'supercategory': 'food'}, {'id': 53, 'name': 'pizza', 'supercategory': 'food'}, {'id': 54, 'name': 'donut', 'supercategory': 'food'}, {'id': 55, 'name': 'cake', 'supercategory': 'food'}, {'id': 56, 'name': 'chair', 'supercategory': 'furniture'}, {'id': 57, 'name': 'couch', 'supercategory': 'furniture'}, {'id': 58, 'name': 'potted_plant', 'supercategory': 'furniture'}, {'id': 59, 'name': 'bed', 'supercategory': 'furniture'}, {'id': 60, 'name': 'dining_table', 'supercategory': 'furniture'}, {'id': 61, 'name': 'toilet', 'supercategory': 'furniture'}, {'id': 62, 'name': 'tv', 'supercategory': 'electronic'}, {'id': 63, 'name': 'laptop', 'supercategory': 'electronic'}, {'id': 64, 'name': 'mouse', 'supercategory': 'electronic'}, {'id': 65, 'name': 'remote', 'supercategory': 'electronic'}, {'id': 66, 'name': 'keyboard', 'supercategory': 'electronic'}, {'id': 67, 'name': 'cell_phone', 'supercategory': 'electronic'}, {'id': 68, 'name': 'microwave', 'supercategory': 'appliance'}, {'id': 69, 'name': 'oven', 'supercategory': 'appliance'}, {'id': 70, 'name': 'toaster', 'supercategory': 'appliance'}, {'id': 71, 'name': 'sink', 'supercategory': 'appliance'}, {'id': 72, 'name': 'refrigerator', 'supercategory': 'appliance'}, {'id': 73, 'name': 'book', 'supercategory': 'indoor'}, {'id': 74, 'name': 'clock', 'supercategory': 'indoor'}, {'id': 75, 'name': 'vase', 'supercategory': 'indoor'}, {'id': 76, 'name': 'scissors', 'supercategory': 'indoor'}, {'id': 77, 'name': 'teddy_bear', 'supercategory': 'indoor'}, {'id': 78, 'name': 'hair_dryer', 'supercategory': 'indoor'}, {'id': 79, 'name': 'toothbrush', 'supercategory': 'indoor'}]
with open('dataset/ODFN/train/annotations/train.json', 'r') as f:
    old_data = json.load(f)
    old_data['categories'] = categories
with open('dataset/ODFN/train/annotations/train.json', 'w') as f:
    # write the new data
    json.dump(old_data, f)