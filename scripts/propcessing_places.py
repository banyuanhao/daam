# read the first colomn of the file /home/banyh2000/diffusion/daam/daam/Scene hierarchy.xlsx and store it in a variable called 'scene_hierarchy'
import pandas as pd
scene_hierarchy = pd.read_excel('/home/banyh2000/diffusion/daam/daam/Scene hierarchy.xlsx', usecols=[0])
# extract the context of scene_hierarchy and store it in a str list variable called 'context'
context = scene_hierarchy['Unnamed: 0'].tolist()
context = [ins.split('/')[-1][:-1].replace('_',' ') for ins in context][1:]
print(context)
context_1 = [f"a photo of {ins}" for ins in context]
context_2 = [f"a picture of {ins}" for ins in context]
context_3 = [f"a image of {ins}" for ins in context]
context = context_1 + context_2 + context_3
for i in range(8):
    with open(f'/home/banyh2000/diffusion/daam/daam/wrapupdata/remove/remove_data/prompts_add{i+1}_places.txt', 'w') as f:
        for line in context[i*len(context)//8:(i+1)*len(context)//8]:
            f.write(line+'\n')