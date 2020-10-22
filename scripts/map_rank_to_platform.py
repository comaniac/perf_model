import tqdm
import glob
import heapq
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import tvm
from tvm.autotvm.record import load_from_file
from lorien.workload import Workload

# Number of top configs.
top_n_cfgs = 32

# The config of the first platform.
platform1_dir = sys.argv[1]

# The config folder of the second platform.
# When the target config is missing in the full log,
# we may re-tune it and put it to another log file.
# In short, when two platform2 dirs are specified,
# we will aggregate their logs as the reference.
platform2_dirs = [sys.argv[2]]
if len(sys.argv) == 4:
    platform2_dirs.append(sys.argv[3])

log_target = None
for log_file in glob.glob('{}/*'.format(platform2_dirs[0])):
    for inp, res in load_from_file(log_file):
        log_target = str(inp.target)
        break
    break
print(log_target)

# Histogram of rank shifting counts
hist = {}

def run_one_wkl(platform1_log, platform2_dirs):
    target_wkl = None
    cfgs = []
    log_files = None
    for inp, res in load_from_file(platform1_log):
        # Update the target string to generate the same SHA2 hash code
        # to identify the full config log file.
        if target_wkl is None:
            inp.task.target = inp.target
            target_wkl = Workload.from_task(inp.task)
            target_wkl['target'] = log_target
    
            # The full config log file name by Lorien is composed of
            # <SHA2>-<5-byte UUID4>.json
            log_files = []
            for platform2_dir in platform2_dirs:
                log_path = '{0}/{1}*.json'.format(platform2_dir, target_wkl.hash_sha2())
                log_files += glob.glob(log_path)
            if not log_files:
                print('Log missing for %s: %s' % (str(target_wkl), target_wkl.hash_sha2()))
                return
    
        # Only focus on the best N configs.
        target_cfg_str = str(inp.config)
        if len(cfgs) < top_n_cfgs:
            heapq.heappush(cfgs, (-np.mean(res.costs), target_cfg_str))
        elif np.mean(res.costs) < -cfgs[0][0]:
            heapq.heappop(cfgs)
            heapq.heappush(cfgs, (-np.mean(res.costs), target_cfg_str))
    
    # Load and sort all configs.
    assert log_files is not None
    all_records = {}
    for log_file in log_files:
        for inp, res in load_from_file(log_file):
            # De-duplication.
            cfg = str(inp.config)
            if cfg not in all_records:
                all_records[cfg] = (inp, res)
            else:
                old_res = all_records[cfg][1]
                all_records[cfg] = (inp, res if np.mean(res.costs) < np.mean(old_res.costs) else old_res)
    all_records = sorted(all_records.values(), key=lambda p: np.mean(p[1].costs))
    cfg_to_rank_on_p2 = {str(inp.config): rank for rank, (inp, _) in enumerate(all_records)}
    mapped = [False for _ in range(len(cfg_to_rank_on_p2) + 1)]
    
    assert target_wkl is not None
    #task = target_wkl.to_task()
    #space_size = np.product([len(v.entities) for v in task.config_space.space_map.values()])
    
    cfgs.sort(key=lambda x: -x[0])
    for rank1, (target_cost, target_cfg_str) in enumerate(cfgs):
        target_cost = -target_cost
    
        #display_name = '{} {} rank1 {}'.format(str(task), target_cfg_str, rank1)
    
        # Map the rank from the first platform to the second platform.
        if target_cfg_str not in cfg_to_rank_on_p2:
            continue
        rank2 = cfg_to_rank_on_p2[target_cfg_str]
        rank_shift = rank1 - rank2
        mapped[rank2] = True
        #print('{:40s}\trank2 {:5d}\t{:5d}\t{:10d}'.format(display_name,
        #      rank2, len(all_records), space_size))
        #print('%d\t%d' % (rank1, rank2))
        if rank_shift not in hist:
            hist[rank_shift] = 0
        hist[rank_shift] += 1

        #if rank_shift <= -2000:
        #    print('{}\trank2 {:5d}\t{:5d}\t{:10d}'.format(display_name, rank2, len(all_records), space_size))

    return

for config_file in tqdm.tqdm(glob.glob('{}/*.json'.format(platform1_dir))):
    run_one_wkl(config_file, platform2_dirs)

with open('out.log', 'w') as filep:
    for rank_shift, cnt in sorted(hist.items(), key=lambda item: item[0]):
        filep.write('%d\t%d\n' % (rank_shift, cnt))
