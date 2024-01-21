python scripts/daamwandb.py --prompt "a couple walk along the riverside in Paris" --negative_prompt "Eiffel tower" --time_id 2 3 3 4 4 5 5 6 --group metric_todaam_del --seed 6463344 --word "couple"

# 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072
# python scripts/ratio.py --prompt "professional office woman" --negative_prompt "glasses" --seed 7056021 --group running --negative_time 4 --bound_box 4 4 20 20

#python scripts/metric.py --prompt "a couple walk along the riverside in Paris" --negative_prompt "Eiffel tower" --look_mode 'pu' --look_part latent --look_time 2 3 4 5 --group metric_binding --seed 6463344 --out_file pics/pic.png --visualizer negative_uncond