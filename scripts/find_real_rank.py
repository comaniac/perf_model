import glob
import os
import sys

import numpy as np

import tvm
from tvm.autotvm.record import load_from_file
from lorien.workload import Workload

# The config found by the rank model to be analyzed
best_config_log = sys.argv[1]

# The full config by AutoTVM to be referenced.
all_log_dir = sys.argv[2]

# The target string in the full config log file.
log_target = sys.argv[3]

wkl_to_log_file = {}
missed = 0
total = 0
for inp, res in load_from_file(best_config_log):
    # Update the target string to generate the same SHA2 hash code
    # to identify the full config log file.
    inp.task.target = inp.target
    target_wkl = Workload.from_task(inp.task)
    target_wkl['target'] = log_target
    target_cfg_str = str(inp.config)

    if target_wkl not in wkl_to_log_file:
        # The full config log file name by Lorien is composed of
        # <SHA2>-<5-byte UUID4>.json
        total += 1
        log_path = '{0}/{1}*.json'.format(all_log_dir, target_wkl.hash_sha2())
        log_files = glob.glob(log_path)
        if not log_files:
            print('Log missing for %s' % str(target_wkl))
            missed += 1
        wkl_to_log_file[target_wkl] = (target_cfg_str, np.mean(res.costs), log_files)

    # Only focus on the best config found by the ranking model.
    if np.mean(res.costs) < wkl_to_log_file[target_wkl][1]:
        wkl_to_log_file[target_wkl] = (target_cfg_str, np.mean(res.costs), wkl_to_log_file[target_wkl][2])

print('{} / {} missed'.format(missed, total))

for idx, (target_wkl, (target_cfg_str, target_cost, log_files)) in enumerate(wkl_to_log_file.items()):
    if not log_files:
        continue

    # Load and sort all configs.
    all_records = []
    for log_file in log_files:
        all_records += list(load_from_file(log_file))
    all_records = sorted(all_records, key=lambda p: np.mean(p[1].costs))

    # Find the real rank of the target config.
    found = False
    for rank, record in enumerate(all_records):
        if target_cfg_str == str(record[0].config):
            found = True
            print('{}\t{}\t{}'.format(record[0].task.name, rank, len(all_records)))
            break
        elif target_cost < np.mean(record[1].costs):
            found = True
            print('{}\t{}*\t{}'.format(record[0].task.name, rank, len(all_records)))
            break
    if not found:
        print('{}\t{}\t{}'.format(record[0].task.name, -1, len(all_records)))

