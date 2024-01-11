export CUDA_VISIBLE_DEVICES=7
# python scripts/ratio.py --prompt "Professional office woman" --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 0 0 64 64 --wandb

# python scripts/ratio.py --prompt "Professional office woman" --negative_prompt "glasses" --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 0 0 64 64 --wandb

# python scripts/ratio.py --prompt "Professional office woman" --negative_prompt "glasses" --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 20 20 24 24 --wandb

# python scripts/ratio.py --prompt "Professional office woman" --negative_prompt "glasses"  --group ratio_del --seed 7995971 --bound_box 20 0 30 10

# python scripts/ratio.py --prompt "a cat running across the yard" --negative_prompt fence --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 20 0 30 10
# python scripts/ratio.py --prompt "a cat running across the yard" --negative_prompt trees --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 20 0 30 10
# python scripts/ratio.py --prompt "A lawn in a courtyard" --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 20 0 30 10 --wandb
# python scripts/ratio.py --prompt "A lawn in a courtyard" --negative_prompt "grass" --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 20 0 30 10 --wandb
# python scripts/ratio.py --prompt "A lawn in a courtyard" --negative_prompt "flowers" --group ratio_del --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --bound_box 20 0 30 10 --wandb

# ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off

# python scripts/daamwandb.py --prompt "a woman sitting in a cafe, smiling" --negative_prompt "unattractive woman" --words "n:unattractive" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "a woman sitting in a cafe, smiling" --negative_prompt "unattractive woman" --words "n:unattractive" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "a stylish woman walking on the street" --negative_prompt "deformity" --words "n:deformity" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb


# ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face

# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "overexposed" --words "n:overexposed" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "underexposed" --words "n:underexposed" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "bad art" --words "n:bad art" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "tiling" --words "n:tiling" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "body out of frame" --words "n:body out of frame" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "body out of framed" --words "n:body" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "body out of framed" --words "n:out" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "body out of framed" --words "n:frame" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "body out of framed" --words "n:out" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "bad anatomy, signature, watermark, username, error, missing limbs, error" --words "n:bad anatomy" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb


# ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "ugly" --words "n:ugly" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "disfigured" --words "n:disfigured" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "deformed" --words "n:deformed" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "bad anatomy" --words "n:bad anatomy" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "distorted face" --words "n:distorted face" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "distorted face" --words "n:distorted" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "distorted face" --words "n:face" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "extra limbs" --words "n:extra limbs" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "extra limbs" --words "n:extra" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "extra limbs" --words "n:limbs" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "ugly" --words "n:ugly" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused girl sits at his desk in a sunny classroom." --negative_prompt "ugly" --words "n:ugly" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused girl sits at his desk in a sunny classroom." --negative_prompt "ugly" --words "girl" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused girl sits at his desk in a sunny classroom." --negative_prompt "ugly girl" --words "n:girl" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "ugly male student" --words "n:student" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "The student is ugly" --words "n:student" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused male student sits at his desk in a sunny classroom." --negative_prompt "The student is ugly" --words "n:ugly" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a woman sitting in a cafe, smiling" --negative_prompt "ugly" --words "n:ugly" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "a stylish woman walking on the street" --negative_prompt "distorted face" --words "n:face" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a stylish woman walking on the street" --negative_prompt "distorted face" --words "n:distorted face" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# Seared medium-rare steak on a board with herbs and visible seasonings.

# python scripts/daamwandb.py --prompt "a piece of medium-rare steak on a board with herbs and visible seasonings." --negative_prompt "rotten" --words "n:rotten" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a piece of medium-rare steak on a board with herbs and visible seasonings." --negative_prompt "badly cooked" --words "n:badly" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a piece of medium-rare steak on a board with herbs and visible seasonings." --negative_prompt "sour" --words "n:sour" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "a piece of medium-rare steak on a board with herbs and visible seasonings." --negative_prompt "soggy" --words "n:soggy" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "a piece of medium-rare steak on a board with herbs and visible seasonings." --negative_prompt "soggy" --words "n:soggy" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# asymmetrical
# badly
# cooked
# black
# and
# white
# blurry
# burnt
# cartoon
# cgi
# cold
# dark
# distorted
# expired
# grainy
# human
# parts
# incomplete
# low
# resolution
# lowres
# moldy
# mutated
# overcooked
# oversaturated
# rotten
# sculpture
# sketch
# soggy
# sour
# stale
# undercooked
# unfocused
# unrealistic
# upside
# down



# python scripts/daamwandb.py --prompt "A focused boy sits at his desk in a sunny classroom." --negative_prompt "ugly" --words "n:ugly" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused boy sits at his desk in a sunny classroom." --negative_prompt "disfigured" --words "n:disfigured" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused boy sits at his desk in a sunny classroom." --negative_prompt "deformed" --words "n:deformed" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused boy sits at his desk in a sunny classroom." --negative_prompt "bad anatomy" --words "n:bad anatomy" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb
# python scripts/daamwandb.py --prompt "A focused boy sits at his desk in a sunny classroom." --negative_prompt "distorted face" --words "n:distorted face" --group research.sh  --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb

# python scripts/daamwandb.py --prompt "Professional office woman wearing glasses" --group constrastive --word glasses --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --wandb --time_id 1 5 6 10 11 15 16 20 21 25 26 30

# python scripts/daamwandb.py --prompt "Professional office woman" --negative_prompt "glasses" --group constrastive --word n:glasses --seed 6463344 7056021 679216 4343903 8577767 8152514 6793668 5088744 7995971 6007072 --time_id 1 5 6 10 11 15 16 20 21 25 26 30 --wandb