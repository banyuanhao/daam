##### description to study the time when shape color texture attend to the right place

# 

#### how to do a quantity study?

#### three types of adjs, color, texture, shape

### only positive prompt is used, one object, daamwandb, time_id
# a blue bench -> blue

python scripts/daamwandb.py --prompt "a blue bench" --words "blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# a blue bench -> bench

python scripts/daamwandb.py --prompt "a blue bench" --words "bench" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, one object, daamwandb, time_id
# a bench : blue -> n:blue
# a bench : blue -> n:bench
# a bench : blue bench -> n:blue
# a bench : blue bench -> n:bench
# a bench : blue bench -> bench

python scripts/daamwandb.py --prompt "a bench" --negative_prompt "blue" --words "n:blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a bench" --negative_prompt "blue" --words "bench" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a bench" --negative_prompt "blue bench" --words "n:blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a bench" --negative_prompt "blue bench" --words "n:bench" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a bench" --negative_prompt "blue bench" --words "bench" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### only positive prompt is used, two objects, one attribute, daamwandb time_id
# a blue bench and a yellow balloon -> blue
# a blue bench and a yellow balloon -> bench
# a blue bench and a yellow balloon -> yellow
# a blue bench and a yellow balloon -> balloon
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "bench" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, two objects, one attribute, daamwandb time_id
# a blue bench and a balloon: blue balloon -> blue
# a blue bench and a balloon: blue balloon -> n:blue
# a blue bench and a balloon: blue balloon -> balloon
# a blue bench and a balloon: blue balloon -> n:balloon
python scripts/daamwandb.py --prompt "a blue bench and a balloon" --negative_prompt "blue balloon" --words "blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a balloon" --negative_prompt "blue balloon" --words "n:blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a balloon" --negative_prompt "blue balloon" --words "balloon" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a balloon" --negative_prompt "blue balloon" --words "n:balloon" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### only positive prompt is used, two objects, two attributes, daamwandb time_id
# a blue bench and a yellow balloon -> blue
# a blue bench and a yellow balloon -> bench
# a blue bench and a yellow balloon -> yellow
# a blue bench and a yellow balloon -> balloon
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "bench" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "yellow" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --words "balloon" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, two objects, two attributes, daamwandb time_id
# a blue bench and a yellow balloon: blue balloon -> blue
# a blue bench and a yellow balloon: blue balloon -> n:blue
# a blue bench and a yellow balloon: blue balloon -> balloon
# a blue bench and a yellow balloon: blue balloon -> n:balloon
# a blue bench and a yellow balloon: blue balloon -> yellow
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --negative_prompt "blue balloon" --words "blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --negative_prompt "blue balloon" --words "n:blue" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --negative_prompt "blue balloon" --words "balloon" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --negative_prompt "blue balloon" --words "n:balloon" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a blue bench and a yellow balloon" --negative_prompt "blue balloon" --words "yellow" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb



### negative prompt only have adj, high related adj
# a little girl running in the park: uncute -> n:uncute
# a little girl running in the park: cute -> n:cute

python scripts/daamwandb.py --prompt "a little girl running in the park" --negative_prompt "uncute, 3d" --words "n:uncute" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a little girl running in the park" --negative_prompt "uncute, 3d" --words "n:3d" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb



### context
## color
# a blue bench and a yellow balloon
# a brown book and a blue clock
# a brown dog and a yellow sheep
# a blue apple and a yellow banana
# a blue shirt and a yellow tie
# a yellow shirt and a blue tie
# A white kitchen counter with a big, brown bowl on it.
# A brown kitchen counter with a big, white bowl on it.
## texture
# a metallic watch and leather gloves
# a plastic toy and a glass jar
# a metallic fork and a wooden spoon
# a wooden table and a glass vase
# a wooden table and a plastic vase
# rubber gloves and a fabric dress
# a plastic bowl and a wooden spoon
# a fabric shirt and a leather bag 
## shape
# a round table and a square chair
# a round table and a square chair
# a cubic box and a cylindrical canister
# a round vase and a rectangular photo frame
# a big bag and a small handbag


########
#### three types of adjs, color, texture, shape
# a wooden table and a glass vase
# a wooden table and a glass vase
### only positive prompt is used, one object, daamwandb, time_id
# a wooden table -> wooden

python scripts/daamwandb.py --prompt "a wooden table" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# a wooden table -> table

python scripts/daamwandb.py --prompt "a wooden table" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, one object, daamwandb, time_id
# a table : wooden -> n:wooden
# a table : wooden -> n:table
# a table : wooden table -> n:wooden
# a table : wooden table -> n:table
# a table : wooden table -> table

python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden table" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden table" --words "n:table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a table" --negative_prompt "wooden table" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### only positive prompt is used, two objects, one attribute, daamwandb time_id
# a wooden table and a glass vase -> wooden
# a wooden table and a glass vase -> table
# a wooden table and a glass vase -> glass
# a wooden table and a glass vase -> vase
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, two objects, one attribute, daamwandb time_id
# a wooden table and a vase: wooden vase -> wooden
# a wooden table and a vase: wooden vase -> n:wooden
# a wooden table and a vase: wooden vase -> vase
# a wooden table and a vase: wooden vase -> n:vase
python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a vase" --negative_prompt "wooden vase" --words "n:vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### only positive prompt is used, two objects, two attributes, daamwandb time_id
# a wooden table and a glass vase -> wooden
# a wooden table and a glass vase -> table
# a wooden table and a glass vase -> glass
# a wooden table and a glass vase -> vase
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "table" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "glass" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --words "vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

### both positive prompt and negative prompt, two objects, two attributes, daamwandb time_id
# a wooden table and a glass vase: wooden vase -> wooden
# a wooden table and a glass vase: wooden vase -> n:wooden
# a wooden table and a glass vase: wooden vase -> vase
# a wooden table and a glass vase: wooden vase -> n:vase
# a wooden table and a glass vase: wooden vase -> glass
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "n:wooden" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "n:vase" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
python scripts/daamwandb.py --prompt "a wooden table and a glass vase" --negative_prompt "wooden vase" --words "glass" --time_id 1 3 4 6 7 9 10 12 13 15 16 18 19 21 22 24 25 27 28 30 --group critical_step_texture  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb


############

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

python scripts/daamwandb.py --prompt "professional office woman" --words woman --time_id 1 3 4 6 --group critical_step_color  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072

