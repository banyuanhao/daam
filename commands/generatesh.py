
seeds = "6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072"
steps = 30
group = "unclip_noun"
token = -1


with open(f'commands/context/{group}.txt', 'r') as f:
    lines = f.read()
    
lines = lines.split('\n')
shs = []
for line in lines:
    prompt, negative_prompt = line.split('.')
    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()
    sh = f"python ~/diffusion/daam/scripts/wrap_up_ratio_unclip.py --prompt \"{prompt}\" --negative_prompt \"{negative_prompt}\" --seed {seeds} --steps {steps} --group {group} --token {token}"
    shs.append(sh)

with open(f'commands/shs/{group}.txt', 'w') as f:
    f.write('\n'.join(shs))
    
