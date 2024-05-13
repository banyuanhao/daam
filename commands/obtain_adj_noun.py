import json

with open('/home/banyh2000/diffusion/daam/wrapupdata/ratio_adj.json') as f:
    data = json.load(f)
    
lines = []
for ins in data:
    lines.append(ins['prompt']+'. '+ins['negative_prompt'])

with open('commands/context/adj_.txt', 'w') as f:
    for line in lines:
        f.write(line+'\n')