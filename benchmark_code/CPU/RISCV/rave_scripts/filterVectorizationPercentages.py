import os
import sys

"""
This script filters the vectorization percentages from the RAVE output files
and moves them to a filtered folder. This should run only once when all the rave
analysis has already run. Cleanup of data should be run by a different script. This
just filters the data (vector length and vector mix) from the Rave output files.
"""

rave_output_dir = "/home/pmpakos/vvrettos-stuff/rave-outputs"
paraver_dir="/home/pmpakos/vvrettos-stuff/wxparaver-4.12.0"
paramedir_bin= os.path.join(paraver_dir, "bin", "paramedir")

# Check if the executable is in the path
if not os.path.exists(paramedir_bin):
    print(f"Executable {paramedir_bin} does not exist. Please check the path.")
    sys.exit(1)

paraver_config_dir='/home/pmpakos/vvrettos-stuff/rave-scripts/paraver_cfgs/rave/per_phase_cfgs'
filtered_dir='/home/pmpakos/vvrettos-stuff/filtered-rave-outputs'

# Find the name of the .prv file
executable_name = "spmv_csr_vector_rave_d.exe"
prv_file = f'rave-{executable_name}.prv'

# List rave-outputs dir
rave_outputs = os.listdir(rave_output_dir)
vector_length_config = "table_average_vl_per_instruction_per_phase.cfg"
vector_mix_config = "tables_vector_mix_per_phase.cfg"

config_files = [vector_length_config, vector_mix_config]

# PWD
current_working_dir = os.getcwd()

config_command = ""
for config in config_files:
    config_command += os.path.join(paraver_config_dir, config) + " "

# Execute paramedir with config files
# For each file: 
# 1. Run the paramedir configs that we are interested in 
# 2. Rename the files to include the matrix name
# 3. Move the output files to a filtered folder

for folder in rave_outputs:
    print("Currently processing folder:", folder)
    print("Matrix Name:", folder.split("spmv_csr_vector_rave_d_")[1])
    prv_path=os.path.join(rave_output_dir, folder, prv_file)
    if not os.path.exists(prv_path):
        print(f"File {prv_path} does not exist. Skipping...")
        continue

    # Get the matrix name from the folder
    matrix_name = folder.split("spmv_csr_vector_rave_d_")[1]
    
    # Create folder in filtered dir
    filtered_folder = os.path.join(filtered_dir, matrix_name)
    if not os.path.exists(filtered_folder):
        os.makedirs(filtered_folder)

    config_type = config.split(".cfg")[0]
    filtered_output_file = os.path.join(filtered_folder, config_type)
    if os.path.exists(filtered_output_file):
        print(f"File {filtered_output_file} already exists. Skipping...")
        continue

    # If size of .prv file more than 25GB, skip execution (will OOM)
    size = os.path.getsize(prv_path)
    if size > 25 * 1024 * 1024 * 1024:
        print(f"File {prv_path} is too large. Skipping...")
        continue

    # Actually execute
    os.system(f"{paramedir_bin} {prv_path} {config_command}")

    print(f"Finished executing paramedir for {matrix_name}")

    # Move the output files to the filtered folder
    for config in config_files:
        config_type = config.split(".cfg")[0]
        local_output_file = os.path.join(current_working_dir, config_type)
        filtered_output_file = os.path.join(filtered_folder, config_type)

        os.rename(local_output_file, filtered_output_file)