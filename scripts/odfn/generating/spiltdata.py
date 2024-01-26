# spilt data into train, val, test
from utils_odfn import seeds_plus
from pathlib import Path
import shutil

class_name = 'baseball_glove'
data_dir = Path('/home/banyh2000/diffusion/daam/dataset/ODFN/version_2/')
train_dir = data_dir / 'train' / 'images' / class_name
val_dir = data_dir / 'val' / 'images' / class_name
test_dir = data_dir / 'test' / 'images' / class_name


train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)

train_seeds = seeds_plus[:17500]
val_seeds = seeds_plus[17500:18500]
test_seeds = seeds_plus[18500:]

# for seed in test_seeds:
#     src = train_dir / str(seed)
#     dst = test_dir
#     # move the folder src to dst
#     shutil.move(str(src), str(dst))
    
for seed in val_seeds:
    src = train_dir / str(seed)
    dst = val_dir
    # move the folder src to dst
    shutil.move(str(src), str(dst))