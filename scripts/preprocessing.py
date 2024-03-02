# import random
# random.seed(0)
# path = '/mnt/data0/banyuanhao/dataset/removing/MSVD.txt'
# base_path = '/home/banyh2000/diffusion/daam/daam/wrapupdata/remove/remove_data'
# # read context from txt file
# with open(path, 'r') as f:
#     lines = f.readlines()
#     f.close()
    
# lines = [' '.join(line.split(' ')[1:]).strip() for line in lines]
# print(lines[:10])
# random.shuffle(lines)
# for i in range(8):
#     line = lines[i*125:125+i*125]
#     with open(base_path+'/prompts_add'+str(i+1)+'_msvd.txt', 'w') as f:
#         f.write('\n'.join(line))
#         f.close()
#将字符串列表合并，并用空格分隔开

# import json
# path = '/mnt/data0/banyuanhao/dataset/vatex.json'
# with open(path, 'r') as f:
#     data = json.load(f)
#     f.close()
    
# lines = []
# for di in data:
#     lines.append(di['enCap'][0])

# base_path = '/home/banyh2000/diffusion/daam/daam/wrapupdata/remove/remove_data'
# for i in range(8):
#     line = lines[i*125:125+i*125]
#     with open(base_path+'/prompts_add'+str(i+1)+'_vatex.txt', 'w') as f:
#         f.write('\n'.join(line))
#         f.close()

import json
import random
path = '/mnt/data0/banyuanhao/dataset/removing/nocap.json'
with open(path, 'r') as f:
    data = json.load(f)
    f.close()

random.seed(0)
annotations = data['annotations']
random.shuffle(annotations)

lines = []
for di in annotations:
    lines.append(di['caption'])

base_path = '/home/banyh2000/diffusion/daam/daam/wrapupdata/remove/remove_data'
for i in range(8):
    line = lines[i*125:125+i*125]
    with open(base_path+'/prompts_add'+str(i+1)+'_nocap.txt', 'w') as f:
        f.write('\n'.join(line))
        f.close()