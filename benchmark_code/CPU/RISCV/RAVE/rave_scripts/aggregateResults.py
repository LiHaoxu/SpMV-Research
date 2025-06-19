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

cwd = os.getcwd()
root_dir = os.path.dirname(cwd)
executable_name ="spmv_csr_vector_rave_d"

# Input path
filtered_rave_outputs=root_dir+f'/filtered-rave-outputs/{executable_name}'
instruction_mix_output_name="tables_vector_mix_per_phase"
vector_length_output_name="table_average_vl_per_instruction_per_phase"

output_file_names = [instruction_mix_output_name, vector_length_output_name]

# Python programmers hate him!
aggregations = {
    instruction_mix_output_name : aggregate_instruction_mix,
    vector_length_output_name : aggregate_vl_per_instruction
}

# Columns: matrix_name, scalar_perc, vectorp_perc, average_vector_length
header = ['matrix_name', 'scalar_perc', 'vector_perc', 'average_vector_length', 'average_non_zeros_per_row']
df = pd.DataFrame(columns=header)

matrix_df = pd.read_csv('matrix_features.csv') # matrix,nr_rows,nr_cols,nr_nzeros

folder_list = os.listdir(filtered_rave_outputs)
print(folder_list)

for folder in folder_list:
    output_file_path = os.path.join(filtered_rave_outputs, folder)

    nr_rows = matrix_df[matrix_df['matrix'] == folder]['nr_rows'].values[0]
    nr_cols = matrix_df[matrix_df['matrix'] == folder]['nr_cols'].values[0]
    nr_nzeros = matrix_df[matrix_df['matrix'] == folder]['nr_nzeros'].values[0]
    non_zero_per_row = nr_nzeros / nr_rows
    non_zero_per_row = matrix_df[matrix_df['matrix'] == folder]['nnz-r-avg'].values[0]
    
    scalar_perc, vector_perc = aggregations[instruction_mix_output_name](os.path.join(output_file_path, instruction_mix_output_name))
    if (scalar_perc == 0):
        continue # Skip this row, it has no data

    average_vector_length = aggregations[vector_length_output_name](os.path.join(output_file_path, vector_length_output_name))
    
    df.loc[len(df)] = [folder, scalar_perc, vector_perc, average_vector_length, non_zero_per_row]

df.to_csv(f"{executable_name}-AggregatedResults.csv", index=False)

# lista = ['olm5000', 'nv2010', 'scircuit', 'mac_econ_fwd500', 'raefsky3', 'rgg_n_2_17_s0', 'bbmat', 'appu', 'mc2depi', 'rma10', 'cop20k_A', 'thermomech_dK', 'webbase-1M', 'cant', 'ASIC_680k', 'roadNet-TX', 'pdb1HYS', 'TSOPF_RS_b300_c3', 'Chebyshev4', 'consph', 'com-Youtube', 'rajat30', 'radiation', 'Stanford_Berkeley', 'shipsec1', 'PR02R', 'CurlCurl_2']
# for folder in lista:
#     row = matrix_df[matrix_df['matrix'] == folder]
#     # print(row[['nr_rows', 'nnz-r-avg', 'nnz-r-std', 'bw-avg', 'skew_coeff', 'num-neigh-avg', 'cross_row_sim-avg']].values[0])
#     nr_rows = row['nr_rows'].values[0]
#     nnz_r_avg = row['nnz-r-avg'].values[0]
#     nnz_r_std = row['nnz-r-std'].values[0]
#     bw_avg = row['bw-avg'].values[0]
#     skew_coeff = row['skew_coeff'].values[0]
#     num_neigh_avg = row['num-neigh-avg'].values[0]
#     cross_row_sim_avg = row['cross_row_sim-avg'].values[0]
#     print(f"{folder}: {nr_rows}_{nnz_r_avg}_{nnz_r_std}_{bw_avg}_{skew_coeff}_{num_neigh_avg}_{cross_row_sim_avg}")
