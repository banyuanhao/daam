python scripts/generateinterwave.py  --prompt "professional office woman" --negative_prompt "glasses" --group onestep  --seed 36515 --negative_time 3 4 5 6 --estimated_time 3 5 7 9 11 13 15 100  --wandb

python scripts/daamwandb.py --prompt "professional office woman" --negative_prompt "glasses" --words "glasses" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group onestep  --seed 36515 --wandb

python scripts/daamwandbinterwave.py --prompt "professional office woman" --negative_prompt "glasses" --words "n:glasses" --time_id 0 2 3 5 6 8 9 11 12 14 15 17 18 20 21 23 24 26 27 29 --group onestep  --seed 36515 --wandb --negative_time 3 4 5 6