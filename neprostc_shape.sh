#### three types of adjs, color, texture, shape
# a big teddy bear and a small toy car
# a big teddy bear and a small toy car
### only positive prompt is used, one object, daamwandb, time_id
# a big teddy bear -> big
# a big teddy bear and a small toy car
# a big teddy bear and a small toy car
python scripts/daamwandb.py --prompt "a big teddy bear" --words "teddy bear" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# a big teddy bear -> teddy bear

python scripts/daamwandb.py --prompt "a big teddy bear" --words "teddy bear" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, one object, daamwandb, time_id
# a teddy bear : big -> n:big
# a teddy bear : big -> n:teddy bear
# a teddy bear : big teddy bear -> n:big
# a teddy bear : big teddy bear -> n:teddy bear
# a teddy bear : big teddy bear -> teddy bear

python scripts/daamwandb.py --prompt "a teddy bear" --negative_prompt "big" --words "n:big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a teddy bear" --negative_prompt "big" --words "teddy bear" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a teddy bear" --negative_prompt "big teddy bear" --words "n:big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a teddy bear" --negative_prompt "big teddy bear" --words "n:teddy bear" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a teddy bear" --negative_prompt "big teddy bear" --words "teddy bear" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### only positive prompt is used, two objects, one attribute, daamwandb time_id
# a big teddy bear and a small toy car -> big
# a big teddy bear and a small toy car -> teddy bear
# a big teddy bear and a small toy car -> small
# a big teddy bear and a small toy car -> toy car
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "teddy bear" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, two objects, one attribute, daamwandb time_id
# a big teddy bear and a toy car: big toy car -> big
# a big teddy bear and a toy car: big toy car -> n:big
# a big teddy bear and a toy car: big toy car -> toy car
# a big teddy bear and a toy car: big toy car -> n:toy car
python scripts/daamwandb.py --prompt "a big teddy bear and a toy car" --negative_prompt "big toy car" --words "big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a toy car" --negative_prompt "big toy car" --words "n:big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a toy car" --negative_prompt "big toy car" --words "toy car" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a toy car" --negative_prompt "big toy car" --words "n:toy car" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### only positive prompt is used, two objects, two attributes, daamwandb time_id
# a big teddy bear and a small toy car -> big
# a big teddy bear and a small toy car -> teddy bear
# a big teddy bear and a small toy car -> small
# a big teddy bear and a small toy car -> toy car
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "teddy bear" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "small" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --words "toy car" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, two objects, two attributes, daamwandb time_id
# a big teddy bear and a small toy car: big toy car -> big
# a big teddy bear and a small toy car: big toy car -> n:big
# a big teddy bear and a small toy car: big toy car -> toy car
# a big teddy bear and a small toy car: big toy car -> n:toy car
# a big teddy bear and a small toy car: big toy car -> small
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --negative_prompt "big toy car" --words "big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --negative_prompt "big toy car" --words "n:big" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --negative_prompt "big toy car" --words "toy car" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --negative_prompt "big toy car" --words "n:toy car" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a big teddy bear and a small toy car" --negative_prompt "big toy car" --words "small" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_shape  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb