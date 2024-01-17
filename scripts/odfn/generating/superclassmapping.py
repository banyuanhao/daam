# file to create a mapping from category id to superclass id
import json
with open('dataset/ODFN/version_1/val/annotations/val.json', 'r') as f:
    data = json.load(f)
    categories = data['categories']
super_dict = {'outdoor': 0, 'indoor': 1, 'vehicle': 2, 'person': 3, 'electronic': 4, 'animal': 5, 'food': 6, 'appliance': 7, 'furniture': 8, 'accessory': 9, 'kitchen': 10, 'sports': 11}
dictionary = {}
for key in categories:
    dictionary[key['id']] = super_dict[key['supercategory']]
    

print(dictionary)
# {0: 3, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5, 23: 5, 24: 9, 25: 9, 26: 9, 27: 9, 28: 9, 29: 11, 30: 11, 31: 11, 32: 11, 33: 11, 34: 11, 35: 11, 36: 11, 37: 11, 38: 11, 39: 10, 40: 10, 41: 10, 42: 10, 43: 10, 44: 10, 45: 10, 46: 6, 47: 6, 48: 6, 49: 6, 50: 6, 51: 6, 52: 6, 53: 6, 54: 6, 55: 6, 56: 8, 57: 8, 58: 8, 59: 8, 60: 8, 61: 8, 62: 4, 63: 4, 64: 4, 65: 4, 66: 4, 67: 4, 68: 7, 69: 7, 70: 7, 71: 7, 72: 7, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 78: 1, 79: 1}