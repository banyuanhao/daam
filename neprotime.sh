export CUDA_VISIBLE_DEVICES=1

# python scripts/daamwandb.py --prompt "a purple orange on the table" --words "purple" --group "time_bias_test1" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a purple orange on the table" --words "orange" --group "time_bias_test1" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 

# python scripts/daamwandb.py --prompt "a red banana on the table" --words "red" --group "time_bias_test2" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 
# python scripts/daamwandb.py --prompt "a red banana on the table" --words "banana" --group "time_bias_test2" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "a yellow orange on the table" --words "yellow" --group "time_bias_test1" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow orange on the table" --words "orange" --group "time_bias_test1" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 

# python scripts/daamwandb.py --prompt "a yellow banana on the table" --words "yellow" --group "time_bias_test2" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 
# python scripts/daamwandb.py --prompt "a yellow banana on the table" --words "banana" --group "time_bias_test2" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb


# python scripts/daamwandb.py --prompt "A neatly arranged dining table with a vase filled with sunflowers on it" --negative_prompt "plate, fork" --words "n:plate" --group time_oblete_1  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 --wandb
# python scripts/daamwandb.py --prompt "A neatly arranged dining table with a vase filled with sunflowers on it" --negative_prompt "plate, fork" --words "n:fork" --group time_oblete_1 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 --wandb
# python scripts/daamwandb.py --prompt "A neatly arranged dining table with a vase filled with sunflowers on it"  --negative_prompt "plate, fork" --words "table" --group time_oblete_1 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 --wandb

# python scripts/daamwandb.py --prompt "a street in Paris in autumn" --words "street" --group remove_time_2 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 --wandb

# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench"  --words "yellow, balloon, blue, bench" --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "yellow bench" --words "n:yellow" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --words "yellow" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --words "blue" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --words "bench" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "blue balloon" --words "n:balloon" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "blue balloon" --words "n:blue" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "blue" --words "n:blue" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "yellow" --words "n:yellow" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "yellow bench" --words "n:bench" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --words "luffy, toy bear, shiny, trophy" --group binding_time_test2 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --negative_prompt "fluffy trophy" --words "n:fluffy" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test2  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --negative_prompt "fluffy trophy" --words "n:trophy" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test2  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --negative_prompt "fluffy trophy" --words "bear" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test2  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --words "bear" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test2  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "red velvet cake, mouthwatering, slice, vibrant colors, natural lighting" --words "cake" --group binding_time_test3  --seed 6463344 7056021 679216 4343903 8577767 --wandb
# python scripts/daamwandb.py --prompt "red velvet cake, mouthwatering, slice, vibrant colors, natural lighting" --negative_prompt "cutlery" --words "n:cutlery" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test3 --seed 6463344 7056021 679216 4343903 8577767 --wandb
# python scripts/daamwandb.py --prompt "red velvet cake, mouthwatering, slice, vibrant colors, natural lighting" --negative_prompt "cutlery" --words "cake" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group binding_time_test3 --seed 6463344 7056021 679216 4343903 8577767 --wandb

# python scripts/daamwandb.py --prompt "In a beach area, visitors are sunbathing and playing volleyball" --words "visitors" --group remove_time_4 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "In a beach area, visitors are sunbathing and playing volleyball" --words "n:coconut tree" --negative_prompt "coconut tree" --group remove_time_4 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "In a beach area, visitors are sunbathing and playing volleyball" --words "visitors" --negative_prompt "coconut tree" --group remove_time_4 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "Portrait photo of a man" --words "man" --group remove_time_5 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "mustache" --words "man" --group remove_time_5 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "mustache" --words "n:mustache" --group remove_time_5 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "old" --words "man" --group remove_time_5 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "old" --words "n:old" --group remove_time_5 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --words "park" --group remove_time_1 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "people" --words "n:people" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "people" --words "park" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "bench" --words "n:bench" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "bench" --words "park" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "slide" --words "n:slide" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "An indoor swimming pool" --words "pool" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_6 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "An indoor swimming pool" --negative_prompt "windows" --words "n:windows" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_6 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "An indoor swimming pool" --negative_prompt "windows" --words "pool" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_6 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb


# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --words "lawn" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --negative_prompt "flowers" --words "lawn" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --negative_prompt "flowers" --words "n:flowers" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --negative_prompt "windows" --words "n:windows" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A mountain village scenery" --words "mountain" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_8 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A mountain village scenery" --negative_prompt "summer" --words "n:summer" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_8 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A mountain village scenery" --negative_prompt "windows" --words "n:windows" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_8 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

#The scenery of the Alps mountains.
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --words "scenery" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --negative_prompt "clouds" --words "n:clouds" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --negative_prompt "lake" --words "n:lake" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --negative_prompt "cottage" --words "n:cottage" --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --group remove_time_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# Eating barbecue in a courtyard.
# python scripts/daamwandb.py --prompt "People are eating barbecue in a courtyard" --words "people" --group remove_time_10 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "People are eating barbecue in a courtyard" --negative_prompt "table" --words "n:table" --group remove_time_10 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "People are eating barbecue in a courtyard" --negative_prompt "tree" --words "n:tree" --group remove_time_10 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# A man in formal attire carrying a briefcase walking on the street.
# python scripts/daamwandb.py --prompt "A man in formal attire carrying a briefcase walking on the street." --words "briefcase" --group remove_time_11 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A man in formal attire carrying a briefcase walking on the street." --negative_prompt "car" --words "n:car" --group remove_time_11 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# train station, watercolor painting
# python scripts/daamwandb.py --prompt "train station, watercolor painting" --words "station" --group remove_time_12 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "train station, watercolor painting" --negative_prompt "train" --words "n:train" --group remove_time_12 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# close-up fashion photo of a smiling boy
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --words "boy" --group remove_time_13 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --negative_prompt "plaid shirt" --words "n:plaid shirt" --group remove_time_13 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --negative_prompt "smile" --words "n:smile" --group remove_time_13 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --negative_prompt "tooth" --words "n:tooth" --group remove_time_13 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# sunset over a city
# python scripts/daamwandb.py --prompt "sunset over a city" --words "sunset" --group remove_time_14 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "sunset over a city" --negative_prompt "red" --words "n:red" --group remove_time_14 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "sunset over a city" --negative_prompt "cloud" --words "n:cloud" --group remove_time_14 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# holding a parade in the street.
# python scripts/daamwandb.py --prompt "people holding a parade in the street" --words "people" --group remove_time_15 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "people holding a parade in the street" --negative_prompt "car" --words "n:car" --group remove_time_15 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# In the meadow of the deep valley, a small stream flows gently
# python scripts/daamwandb.py --prompt "In the meadow of the deep valley, a small stream flows gently" --words "stream" --group remove_time_16 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "In the meadow of the deep valley, a small stream flows gently" --negative_prompt "tree" --words "n:tree" --group remove_time_16 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# A couple walking along the riverbank in Paris
# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --words "couple" --group remove_time_17 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "tree" --words "n:tree" --group remove_time_17 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "Eiffel Tower" --words "n:Eiffel Tower" --group remove_time_17 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

#Perched on a lush hill, the ancient castle boasted ivy-clad stone walls and towering turrets. Its arched windows overlooked a serene moat, reflecting the grandeur of this historical monument.
# --prompt "Perched on a lush hill, the ancient castle boasted ivy-clad stone walls and towering turrets. historical monument." --words "castle" --group remove_time_18 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# professional office woman
# python scripts/daamwandb.py --prompt "professional office woman" --words "woman" --group remove_time_19 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "professional office woman" --negative_prompt "glasses" --words "n:glasses" --group remove_time_19 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "professional office woman" --negative_prompt "books" --words "n:books" --group remove_time_19 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# A beautiful digital illustration painting of a detailed gothic fantasy valley and forest.
# python scripts/daamwandb.py --prompt "A beautiful digital illustration painting of a detailed gothic fantasy valley and forest." --words "forest" --group remove_time_20 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "a boy wearing glasses" --words "glasses" --group remove_time_21 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed  7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris, with the Eiffel Tower not far away" --words "Eiffel Tower" --group remove_time_22 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "Eiffel Tower" --words "n:Eiffel Tower"  --group metric1 --time_id 2 2 5 5 15 15 25 25 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "Eiffel Tower" --words "couple"  --group metric1 --time_id 2 2 5 5 15 15 25 25 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

python scripts/daamwandb.py --prompt "professional office woman" --negative_prompt "glasses" --words "woman"  --group metric1 --time_id 2 2 5 5 15 15 25 25 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

python scripts/daamwandb.py --prompt "professional office woman" --negative_prompt "glasses" --words "n:glasses"  --group metric1 --time_id 2 2 5 5 15 15 25 25 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb