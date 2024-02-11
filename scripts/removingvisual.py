import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
#  add args package to configure the seed in command line
parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
seed = args.seed

# Define the file paths of the six PNG images
file_paths = [
    f"pics/removing/glasses/no_negative/{seed}.png",
    f"pics/removing/glasses/negative/{seed}.png",
    f"pics/removing/glasses/baseline/{seed}.png",
    f"pics/removing/glasses/negative_6_12/{seed}.png",
    f"pics/removing/glasses/negative_4_12/{seed}.png",
    f"pics/removing/glasses/negative_6_15/{seed}.png"
]
# extract no_negative, negative, baseline, negative_6_12, negative_4_12, negative_5_15, negative_6_15 from file_paths
file_name = [i.split('/')[-2] for i in file_paths]

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(3, 3)

# Loop through the file paths and plot each image on a subplot
for i, file_path in enumerate(file_paths):
    img = mpimg.imread(file_path)
    axs[i // 3, i % 3].imshow(img)
    axs[i // 3, i % 3].axis('off')
    axs.flat[i].set_title(file_name[i])

# Save the figure
plt.savefig(f"pics/removing/glasses/pic.png")
