import json
from utils_odfn import categories,coco_classes_dict,coco_classes

# open the json file and read the data
with open('dataset/coco/annotations/instances_val2014.json', 'r') as f:
    data = json.load(f)
    categories_old = data['categories']
    
print(categories_old)
annotations = []
for dictionary in categories_old:
    name = dictionary["name"].replace(' ', '_')
    if name in coco_classes:
        tmp_dictionary = {
            "id": coco_classes_dict[name],
            "name": name,
            "supercategory": dictionary['supercategory']
        }
        annotations.append(tmp_dictionary)
    else:
        print(name)
        
# print(annotations)
with open('dataset/ODFN/train/annotations/train.json', 'r') as f:
    old_data = json.load(f)
    old_data['categories'] = categories
with open('dataset/ODFN/train/annotations/train.json', 'w') as f:
    # write the new data
    json.dump(old_data, f)