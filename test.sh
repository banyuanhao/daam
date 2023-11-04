echo "Hello World"

# python testdaam.py --prompt "a handsome young man running in a park" \
#                     --negative_prompt "ugly, missing legs" \
#                     --words man --words ugly --words missing_legs --words missing --words legs \
#                     --seed 4588 

# python testdaam.py --prompt "A dog running across the yard with grass" \
#                     --words dog --words grass\
#                     --seed 1845

# python testdaam.py --prompt "A dog running across the yard without grass" \
#                     --words dog --words grass --words without \
#                     --seed 1845

# python testdaam.py --prompt "A dog running across the yard" \
#                     --negative_prompt "grass" \
#                     --words dog --words grass \
#                     --seed 1845

python testdaam.py --prompt "a young man standing right next to a red tesla roadster" \
                     --negative_prompt "ugly guy" \
                     --words ugly_guy \
                     --seed 2345

# python testdaam.py --prompt "a young man standing right next to a red tesla roadster" \
#                      --negative_prompt "handsome guy" \
#                      --words handsome_guy \
#                      --seed 2345
