import pandas as pd
import os

path ='/mnt/data0/banyuanhao/dataset/Image_Labels_Subset_Train_GCC-Labels-training.tsv'

# read the first 1000 lines of the file
lines = []
with open(path, 'r') as f:
    for i in range(1000):
        lines.append(f.readline())
        
lines = [line.split('\t')[0] for line in lines]

for j in range(10):
    with open(os.path.expanduser(f'~/diffusion/daam/daam/prompts_add{j+1}_cc.txt'), 'w') as f:
        for line in lines[j*100:(j+1)*100]:
            f.write(line+'\n')