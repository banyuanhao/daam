from pathlib import Path
import random
import shutil
import argparse

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--experiment_id', type=str, required=True)
args = parser.parse_args()

experiment_id = args.experiment_id


folder_path = Path('experiment') /str(experiment_id)
if not folder_path.is_dir():
    folder_path.mkdir(parents=True, exist_ok=True)

source_file = "commands_generator.py"
destination_file = folder_path/source_file
shutil.copy2(source_file, destination_file)

if not folder_path.is_dir():
    folder_path.mkdir(parents=True, exist_ok=True)



# Define the base command and parameters
base_command = "python ../../scripts/testdaam_time_and_layer.py"
prompt = "a young female, highlights in hair, sitting outside restaurant, brown eyes, wearing a dress, side light"
negative_prompt_words = ["Attractive","Gorgeous","Stunning","Lovely","Handsome","Pretty","Charming","Exquisite","Elegant","Alluring"]

seeds = [random.randint(1, 100000) for _ in range(10)]

# The filename for the output text file
output_file = folder_path/"commands.sh"

# Open the file for writing
with open(output_file, "w") as file:
    file.write("export CUDA_VISIBLE_DEVICES=\"4\"\n")
    # Loop through each pair of time_ids and write the command to the file
    for negative_prompt in negative_prompt_words:
        for seed in seeds:
            # Construct the command
            tmp = prompt.replace('young',negative_prompt)
            command = (
                f"{base_command} --prompt \"{tmp}\" \\\n"
                f"                      --negative_prompt ugly --seed {seed} --experiment_id {experiment_id} \\\n"
                f"                      --words {negative_prompt} --layer_id 0 5 6 10 11 15\n"
            )
            # Write the command to the file
            file.write(command)
    
    file.write(f"#\n seeds = {seeds}\n")


print(f"Commands have been written to {output_file}")