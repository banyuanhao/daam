import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os 
#  add args package to configure the seed in command line
parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--negative_prompt', type=str, default=None)
args = parser.parse_args()
seed = args.seed
negative_prompt = args.negative_prompt

base_path = "/mnt/data0/banyuanhao/dataset/removing/"+negative_prompt+'/'

class_names = os.listdir(base_path)

class_names = [class_name for class_name in class_names if 'json' not in class_name]
# Define the file paths of the six PNG images
file_paths = [
    base_path + class_name +'/' +str(seed)+'.png' for class_name in class_names
]

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3)

# set the size of the figure
fig.set_size_inches(30, 20)
# set the h v space of the figure
plt.subplots_adjust(hspace=0.1, wspace=0.1)  

# Loop through the file paths and plot each image on a subplot
for i, file_path in enumerate(file_paths):
    img = mpimg.imread(file_path)
    axs[i // 3, i % 3].imshow(img)
    axs[i // 3, i % 3].axis('off')
    axs.flat[i].set_title(class_names[i])

# Save the figure
plt.savefig(f"pics/pic.png")
