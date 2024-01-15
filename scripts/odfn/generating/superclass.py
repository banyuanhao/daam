import json
# open the json file and read the data
with open('dataset/ODFN/val/annotations/val.json', 'r') as f:
    data = json.load(f)
    categories = data['categories']
super_dict = {'outdoor': 0, 'indoor': 1, 'vehicle': 2, 'person': 3, 'electronic': 4, 'animal': 5, 'food': 6, 'appliance': 7, 'furniture': 8, 'accessory': 9, 'kitchen': 10, 'sports': 11}

dictionary = {}
for key in categories:
    dictionary[key['id']] = super_dict[key['supercategory']]
    

print(len(dictionary))