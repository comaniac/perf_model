import glob
import heapq
import os
import sys

import numpy as np

import tvm
from tvm.autotvm.record import load_from_file
from lorien.workload import Workload

# Number of top configs.
top_n_cfgs = 5000

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
        wkl_to_log_file[target_wkl] = []

    # Only focus on the best N configs found by the ranking model.
    if len(wkl_to_log_file[target_wkl]) < top_n_cfgs:
        heapq.heappush(wkl_to_log_file[target_wkl], (-np.mean(res.costs), target_cfg_str, log_files))
    elif np.mean(res.costs) < -wkl_to_log_file[target_wkl][0][0]:
        item = heapq.heappop(wkl_to_log_file[target_wkl])
        heapq.heappush(wkl_to_log_file[target_wkl], (-np.mean(res.costs), target_cfg_str, item[2]))

print('{} / {} missed'.format(missed, total))

for idx, (target_wkl, heap) in enumerate(wkl_to_log_file.items()):
    heap.sort(key=lambda x: -x[0])
    for top, (target_cost, target_cfg_str, log_files) in enumerate(heap):
        target_cost = -target_cost

        if not log_files:
            continue
        task = target_wkl.to_task()
        space_size = np.product([len(v.entities) for v in task.config_space.space_map.values()])
        display_name = '{} {} top {}'.format(task.name, task.args[0][1], top)
        display_cost = target_cost * 1e6
    
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
                print('{:40s}\t{:7f}\t{:5d}\t{:5d}\t{:10d}'.format(display_name, display_cost, rank, len(all_records), space_size))
                break
            #elif target_cost < np.mean(record[1].costs):
            #    found = True
            #    print('{:40s}\t{:7f}\t{:5d} *\t{:5d}\t{:10d}\t{:.2f}'.format(
            #        display_name, display_cost, rank, len(all_records), space_size, 100.0 * len(all_records) / space_size))
            #    break
        #if not found:
        #    print('{:40s}\t{:7f}\t{:5d}\t{:5d}\t{}'.format(display_name, display_cost, -1, len(all_records), log_files))

