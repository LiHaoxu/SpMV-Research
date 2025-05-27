import pandas as pd
import os

def aggregate_instruction_mix(filename) -> tuple[float, float]:
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Skipping...")
        return (0, 0)
    fd = open(filename, 'r')

    num_threads = 0
    scalar_perc = 0
    vector_perc = 0
    for line in fd:
        if "scalar" in line:
            scalar_perc += float(line.split()[1])
            num_threads += 1
        elif "vector" in line:
            vector_perc += float(line.split()[1])
    
    # Aggregate scalar/vector
    return (scalar_perc/num_threads, vector_perc/num_threads)

def aggregate_vl_per_instruction(filename) -> float:
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Skipping...")
        return 0

    fd = open(filename, 'r')

    for line in fd:
        if "vector" in line:
            stripped_line = line.split()
            return round(float(stripped_line[2]) / 8.0, 3)

# Input path
filtered_rave_outputs="/home/pmpakos/vvrettos-stuff/filtered-rave-outputs"
instruction_mix_output_name="tables_vector_mix_per_phase"
vector_length_output_name="table_average_vl_per_instruction_per_phase"
validation_matrices_path="/home/pmpakos/vvrettos-stuff/SpMV-Research/validation_matrices"

output_file_names = [instruction_mix_output_name, vector_length_output_name]

# Python programmers hate him!
aggregations = {
    instruction_mix_output_name : aggregate_instruction_mix,
    vector_length_output_name : aggregate_vl_per_instruction
}

# Columns: matrix_name, scalar_perc, vectorp_perc, average_vector_length
header = ['matrix_name', 'scalar_perc', 'vector_perc', 'average_vector_length', 'average_non_zero_per_rows']
df = pd.DataFrame(columns=header)

folder_list = os.listdir(filtered_rave_outputs)

for folder in folder_list:
    output_file_path = os.path.join(filtered_rave_outputs, folder)
    matrix_path = os.path.join(validation_matrices_path, folder + ".mtx")

    # Open file and read up until first line with an integer as first element
    with open(matrix_path, 'r') as file:
        for line in file:
            if line.split()[0].isdigit():
                break
        non_zero_per_row = float(line.split()[2]) / float(line.split()[0])
    
    scalar_perc, vector_perc = aggregations[instruction_mix_output_name](os.path.join(output_file_path, instruction_mix_output_name))
    if (scalar_perc == 0):
        continue # Skip this row, it has no data

    average_vector_length = aggregations[vector_length_output_name](os.path.join(output_file_path, vector_length_output_name))
    df.loc[len(df)] = [folder, scalar_perc, vector_perc, average_vector_length, non_zero_per_row]

df.to_csv("AggregatedVectorizationResults.csv", index=False)