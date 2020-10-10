import argparse
import os
import re
import numpy as np
import glob


parser = argparse.ArgumentParser(description='Read the average time taken for tuning workloads')
parser.add_argument('--dir_path', type=str)
args = parser.parse_args()

all_workload_times = []
for file_path in sorted(glob.glob(os.path.join(args.dir_path, '*.txt'))):
    with open(file_path, 'r') as in_f:
        out = re.findall(r'GFLOPS at Top \d+ | ([\d.]+)s', in_f.read())
        out = [float(ele) for ele in out if len(ele) > 0]
        all_workload_times.extend(out)
mean_workload_time = np.mean(all_workload_times)
print(all_workload_times)
print('Average: ', mean_workload_time)
