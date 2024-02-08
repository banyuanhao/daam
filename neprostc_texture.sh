# #### three types of adjs, color, texture, shape
# # a wooden table and a glass vase
# # a wooden table and a glass vase
# ### only positive prompt is used, one object, daamwandb, time_id
# # a wooden table -> wooden

# python scripts/daamwandb.py --prompt "a wooden table" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# # a wooden table -> table

# python scripts/daamwandb.py --prompt "a wooden table" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# ### both positive prompt and negative prompt, one object, daamwandb, time_id
# # a table : wooden -> n:wooden
# # a table : wooden -> n:table
# # a table : wooden table -> n:wooden
# # a table : wooden table -> n:table
# # a table : wooden table -> table

# python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden table" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden table" --words "n:table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden table" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# ### only positive prompt is used, two objects, one attribute, daamwandb time_id
# # a wooden table and a glass vase -> wooden
# # a wooden table and a glass vase -> table
# # a wooden table and a glass vase -> glass
# # a wooden table and a glass vase -> vase
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# ### both positive prompt and negative prompt, two objects, one attribute, daamwandb time_id
# # a wooden table and a vase: wooden vase -> wooden
# # a wooden table and a vase: wooden vase -> n:wooden
# # a wooden table and a vase: wooden vase -> vase
# # a wooden table and a vase: wooden vase -> n:vase
# python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "n:vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# ### only positive prompt is used, two objects, two attributes, daamwandb time_id
# # a wooden table and a glass vase -> wooden
# # a wooden table and a glass vase -> table
# # a wooden table and a glass vase -> glass
# # a wooden table and a glass vase -> vase
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "glass" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# ### both positive prompt and negative prompt, two objects, two attributes, daamwandb time_id
# # a wooden table and a glass vase: wooden vase -> wooden
# # a wooden table and a glass vase: wooden vase -> n:wooden
# # a wooden table and a glass vase: wooden vase -> vase
# # a wooden table and a glass vase: wooden vase -> n:vase
# # a wooden table and a glass vase: wooden vase -> glass
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "n:vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "glass" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb