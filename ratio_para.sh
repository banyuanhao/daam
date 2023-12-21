export CUDA_VISIBLE_DEVICES=7

# python scripts/ratio.py --prompt "a woman sitting in a cafe" --negative_prompt "amputation" --seed 6463344 --group ratio_gen --bound_box 20 5 40 50

# python scripts/ratio.py --prompt "a yellow balloon on a blue bench" --negative_prompt "yellow bench" --seed 4343903 --group ratio_gen --bound_box 0 30 60 30

# python scripts/ratio.py --prompt "a woman sitting in a cafe" --negative_prompt "huge eyes" --seed 6463344 --group ratio_gen --bound_box 30 12 15 6

# python scripts/ratio.py --prompt "people holding a parade in the street" --negative_prompt "car" --seed 6463344  --group rat --bound_box 20 10 40 40

# python scripts/ratio.py --prompt "red velvet cake, mouthwatering, slice, vibrant colors, natural lighting" --negative_prompt "cutlery" --seed 6463344 --group rat --bound_box 20 10 40 40

python scripts/ratio.py --prompt "A classroom" --negative_prompt "student" --seed 6463344 --group rat --bound_box 20 10 40 40
# python scripts/ratio.py --prompt "a yellow balloon on a blue bench" --negative_prompt "yellow bench" --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --group ratio_binding --wandb
