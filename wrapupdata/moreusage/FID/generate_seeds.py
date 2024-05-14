import json
import random
seeds = random.sample(range(10000000), 1000)
with open('seeds.json', 'w') as f:
    json.dump(seeds, f)