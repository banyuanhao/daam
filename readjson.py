# read the annotation json file of coco dataset, the path is dataset/coco/annotations/instances_train2017.json
# return the list of image id and the list of category id

import json

json_path = 'dataset/coco/annotations/instances_train2014.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# print(data.keys())
# print(data['annotations'][0].keys())
print(data['categories'])
 
# image_id = []
# category_id = []
# for i in range(len(data['annotations'])):
#     image_id.append(data['annotations'][i]['image_id'])
#     category_id.append(data['annotations'][i]['category_id'])
