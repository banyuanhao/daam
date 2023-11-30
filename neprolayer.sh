export CUDA_VISIBLE_DEVICES=3

# python scripts/daamwandb.py --prompt "a purple orange on the table" --words "purple" --group "layer_bias_test1" --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a purple orange on the table" --words "orange" --group "layer_bias_test1" --layer_id 1 3 4 6 7 9 10 12 13 15  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 

# python scripts/daamwandb.py --prompt "a yellow orange on the table" --words "yellow" --group "layer_bias_test1" --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow orange on the table" --words "orange" --group "layer_bias_test1" --layer_id 1 3 4 6 7 9 10 12 13 15  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 

# python scripts/daamwandb.py --prompt "a red banana on the table" --words "red" --group "layer_bias_test2" --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a red banana on the table" --words "banana" --group "layer_bias_test2" --layer_id 1 3 4 6 7 9 10 12 13 15  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 

# python scripts/daamwandb.py --prompt "a yellow banana on the table" --words "yellow" --group "layer_bias_test2" --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow banana on the table" --words "banana" --group "layer_bias_test2" --layer_id 1 3 4 6 7 9 10 12 13 15  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb 

# python scripts/daamwandb.py --prompt "A neatly arranged dining table with a vase filled with sunflowers on it" --negative_prompt "plate, fork" --words "n:plate" --group remove_layer_1  --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 --wandb
# python scripts/daamwandb.py --prompt "A neatly arranged dining table with a vase filled with sunflowers on it" --negative_prompt "plate, fork" --words "n:fork" --group remove_layer_1 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 --wandb

# python scripts/daamwandb.py --prompt "a street in Paris in autumn" --words "street" --group remove_layer_2 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 --wandb

# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench"  --words "yellow, balloon, blue, bench" --group binding_layer_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "yellow bench" --words "n:yellow" --layer_id 1 3 4 6 7 9 10 12 13 15 --group binding_layer_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a yellow balloon on a blue bench" --negative_prompt "yellow bench" --words "n:bench" --layer_id 1 3 4 6 7 9 10 12 13 15 --group binding_layer_test1  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --words "luffy, toy bear, shiny, trophy" --group binding_layer_test2 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --negative_prompt "fluffy trophy" --words "n:fluffy" --layer_id 1 3 4 6 7 9 10 12 13 15 --group binding_layer_test2  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A fluffy toy bear near a shiny trophy" --negative_prompt "fluffy trophy" --words "n:trophy" --layer_id 1 3 4 6 7 9 10 12 13 15 --group binding_layer_test2  --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "red velvet cake, mouthwatering, slice, vibrant colors, natural lighting" --words "cake" --group binding_layer_test3  --seed 6463344 7056021 679216 4343903 8577767 --wandb
# python scripts/daamwandb.py --prompt "red velvet cake, mouthwatering, slice, vibrant colors, natural lighting" --negative_prompt "cutlery" --words "n:cutlery" --layer_id 1 3 4 6 7 9 10 12 13 15 --group binding_layer_test3 --seed 6463344 7056021 679216 4343903 8577767 --wandb
# python scripts/daamwandb.py --prompt "red velvet cake, mouthwatering, slice, vibrant colors, natural lighting" --negative_prompt "cutlery" --words "cake" --layer_id 1 3 4 6 7 9 10 12 13 15 --group binding_layer_test3 --seed 6463344 7056021 679216 4343903 8577767 --wandb

# python scripts/daamwandb.py --prompt "In a beach area, visitors are sunbathing and playing volleyball" --words "visitors" --group remove_layer_4 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "In a beach area, visitors are sunbathing and playing volleyball" --words "n:coconut tree" --negative_prompt "coconut tree" --group remove_layer_4 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "In a beach area, visitors are sunbathing and playing volleyball" --words "visitors" --negative_prompt "coconut tree" --group remove_layer_4 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "Portrait photo of a man" --words "man" --group remove_layer_5 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "mustache" --words "man" --group remove_layer_5 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "mustache" --words "n:mustache" --group remove_layer_5 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "old" --words "man" --group remove_layer_5 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "Portrait photo of a man" --negative_prompt "old" --words "n:old" --group remove_layer_5 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --words "park" --group remove_layer_1 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "people" --words "n:people" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "people" --words "park" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "bench" --words "n:bench" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "bench" --words "park" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "The weather is clear and sunny, at the park." --negative_prompt "slide" --words "n:slide" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_1 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "An indoor swimming pool" --words "pool" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_6 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "An indoor swimming pool" --negative_prompt "windows" --words "n:windows" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_6 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "An indoor swimming pool" --negative_prompt "windows" --words "pool" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_6 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb


# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --words "lawn" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --negative_prompt "flowers" --words "lawn" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --negative_prompt "flowers" --words "n:flowers" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A lawn in a courtyard." --negative_prompt "windows" --words "n:windows" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_7 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A mountain village scenery" --words "mountain" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_8 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A mountain village scenery" --negative_prompt "summer" --words "n:summer" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_8 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A mountain village scenery" --negative_prompt "windows" --words "n:windows" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_8 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# The scenery of the Alps mountains.
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --words "scenery" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --negative_prompt "clouds" --words "n:clouds" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --negative_prompt "lake" --words "n:lake" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "The scenery of the Alps mountains." --negative_prompt "cottage" --words "n:cottage" --layer_id 1 3 4 6 7 9 10 12 13 15 --group remove_layer_9 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# Eating barbecue in a courtyard.
# python scripts/daamwandb.py --prompt "People are eating barbecue in a courtyard" --words "people" --group remove_layer_10 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "People are eating barbecue in a courtyard" --negative_prompt "table" --words "n:table" --group remove_layer_10 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "People are eating barbecue in a courtyard" --negative_prompt "tree" --words "n:tree" --group remove_layer_10 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# A man in formal attire carrying a briefcase walking on the street.
# python scripts/daamwandb.py --prompt "A man in formal attire carrying a briefcase walking on the street." --words "briefcase" --group remove_layer_11 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A man in formal attire carrying a briefcase walking on the street." --negative_prompt "car" --words "n:car" --group remove_layer_11 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# train station, watercolor painting
# python scripts/daamwandb.py --prompt "train station, watercolor painting" --words "station" --group remove_layer_12 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "train station, watercolor painting" --negative_prompt "train" --words "n:train" --group remove_layer_12 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# close-up fashion photo of a smiling boy
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --words "boy" --group remove_layer_13 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --negative_prompt "plaid shirt" --words "n:plaid shirt" --group remove_layer_13 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --negative_prompt "smile" --words "n:smile" --group remove_layer_13 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "close-up fashion photo of a smiling boy" --negative_prompt "tooth" --words "n:tooth" --group remove_layer_13 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# sunset over a city
# python scripts/daamwandb.py --prompt "sunset over a city" --words "sunset" --group remove_layer_14 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "sunset over a city" --negative_prompt "red" --words "n:red" --group remove_layer_14 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "sunset over a city" --negative_prompt "cloud" --words "n:cloud" --group remove_layer_14 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# holding a parade in the street.
# python scripts/daamwandb.py --prompt "people holding a parade in the street" --words "people" --group remove_layer_15 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "people holding a parade in the street" --negative_prompt "car" --words "n:car" --group remove_layer_15 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# In the meadow of the deep valley, a small stream flows gently
# python scripts/daamwandb.py --prompt "In the meadow of the deep valley, a small stream flows gently" --words "stream" --group remove_layer_16 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "In the meadow of the deep valley, a small stream flows gently" --negative_prompt "tree" --words "n:tree" --group remove_layer_16 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# A couple walking along the riverbank in Paris
# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --words "couple" --group remove_layer_17 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "tree" --words "n:tree" --group remove_layer_17 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A couple walking along the riverbank in Paris" --negative_prompt "Eiffel Tower" --words "n:Eiffel Tower" --group remove_layer_17 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# Perched on a lush hill, the ancient castle boasted ivy-clad stone walls and towering turrets. Its arched windows overlooked a serene moat, reflecting the grandeur of this historical monument.
# --prompt "Perched on a lush hill, the ancient castle boasted ivy-clad stone walls and towering turrets. historical monument." --words "castle" --group remove_layer_18 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# professional office woman
# python scripts/daamwandb.py --prompt "professional office woman" --words "woman" --group remove_layer_19 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "professional office woman" --negative_prompt "glasses" --words "n:glasses" --group remove_layer_19 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "professional office woman" --negative_prompt "books" --words "n:books" --group remove_layer_19 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# A beautiful digital illustration painting of a detailed gothic fantasy valley and forest.
# python scripts/daamwandb.py --prompt "A beautiful digital illustration painting of a detailed gothic fantasy valley and forest." --words "forest" --group remove_layer_20 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "a boy wearing glasses" --words "glasses" --group remove_layer_21 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed  7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

python scripts/daamwandb.py --prompt "girl wearing glasses" --words "girl" --group remove_layer_22 --layer_id 1 3 4 6 7 9 10 12 13 15 --seed  7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb