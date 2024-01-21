
export CUDA_VISIBLE_DEVICES=1
# python scripts/generate.py --prompt "professional office woman" --negative_prompt "glasses" --group one_step_estimation --seed 7995971 --negative_time 3 5 7 9 11 13 15 --estimated_time 3 5 7 9 11 13 15

# python scripts/generate.py --prompt "professional office woman" --negative_prompt "glasses" --group one_step_estimation --seed 7995971 --negative_time 3 5 7 9 11 13 15 --estimated_time 15 17 19 21 23 25 --wandb
# python scripts/generate.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "Eiffel Tower" --group one_step_estimation --seed 6793668 --negative_time 3 5 7 9 11 13 15 --estimated_time 3 5 7 9 11 13 15 --wandb
# python scripts/generate.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "Eiffel Tower" --group one_step_estimation --seed 6793668 --negative_time 3 5 7 9 11 13 15 --estimated_time 15 17 19 21 23 25 --wandb


# 15 17 19 21 23 25

# python scripts/daamwandb.py --prompt "train station, watercolor painting" --negative_prompt "train" --words "n:train" --group remove_time_12 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/generate.py --prompt "a yellow balloon on a blue bench"   --negative_prompt "yellow bench" --group one_step_estimation  --seed 679216 --wandb --negative_time 3 5 7 9 11 13 15 --estimated_time 3 5 7 9 11 13 15

python scripts/generate.py --prompt "a yellow balloon on a blue bench"   --negative_prompt "yellow bench" --group one_step_estimation  --seed 4343903 --wandb --negative_time 0 1 2 3 --estimated_time 2 3 4 10

# python scripts/generate.py --prompt "a woman sitting in a cafe" --negative_prompt "unreal" --group one_step_estimation  --seed 679216 --wandb --negative_time 3 5 7 9 11 13 15 --estimated_time 3 5 7 9 11 13 15

# python scripts/generate.py --prompt "a woman sitting in a cafe" --negative_prompt "unreal" --group one_step_estimation  --seed 679216 --wandb --negative_time 3 5 7 9 11 13 15 --estimated_time 15 17 19 21 23 25

# python scripts/generate.py --prompt "A fluffy toy bear near a shiny trophy" --negative_prompt "fluffy trophy" --group one_step_estimation  --seed 6463344 --wandb --negative_time 3 5 7 9 11 13 15 --estimated_time 3 5 7 9 11 13 15

# python scripts/generate.py --prompt "A fluffy toy bear near a shiny trophy" --negative_prompt "fluffy trophy" --group one_step_estimation  --seed 6463344 --wandb --negative_time 3 5 7 9 11 13 15 --estimated_time 15 17 19 21 23 25

# python scripts/generate.py --prompt "a white horse and a brown fence" --negative_prompt "white fence" --group one_step_estimation  --seed 6463344 --wandb --negative_time 3 5 7 9 11 13 15 --estimated_time 3 5 7 9 11 13 15

# python scripts/generate.py --prompt "a white horse and a brown fence" --negative_prompt "white fence" --group one_step_estimation  --seed 6463344 --wandb --negative_time 3 5 7 9 11 13 15 --estimated_time 15 17 19 21 23 25