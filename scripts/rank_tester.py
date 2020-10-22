import glob
import heapq
import math
import os
import sys
import tqdm

import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm.record import load_from_file
from tvm.autotvm.measure import create_measure_batch, MeasureInput
from tvm.autotvm.tuner.callback import log_to_file
from lorien.workload import Workload, gen_log_file_name_from_workload

# Number of top configs.
top_n_cfgs = 32

# The config of the platform.
platform1_dir = sys.argv[1]

platform2_dir = sys.argv[2]

log_target = sys.argv[3]
target = tvm.target.create(log_target)

wkls = {}

def parse_one_log(best_config_log, new_log_dir):
    target_wkl = None
    for inp, res in load_from_file(best_config_log):
        # Update the target string to generate the right SHA2 hash code.
        if target_wkl is None:
            inp.task.target = inp.target
            target_wkl = Workload.from_task(inp.task)
            target_wkl['target'] = log_target
            if target_wkl not in wkls:
                new_log_file_name = gen_log_file_name_from_workload(target_wkl)
                new_log_path = '{0}/{1}'.format(new_log_dir, new_log_file_name)
                wkls[target_wkl] = (new_log_path, [])

        if res.error_no != 0:
            continue
    
        # Only focus on the best N configs.
        new_inp = MeasureInput(target=target, task=inp.task, config=inp.config)
        if len(wkls[target_wkl][1]) < top_n_cfgs:
            heapq.heappush(wkls[target_wkl][1], (-np.mean(res.costs), new_inp))
        elif np.mean(res.costs) < -wkls[target_wkl][1][0][0]:
            heapq.heappop(wkls[target_wkl][1])
            heapq.heappush(wkls[target_wkl][1], (-np.mean(res.costs), new_inp))


def run_one_wkl(wkl, new_log_path, inputs):
    task = wkl.to_task()

    # Re-tune the best configs.
    log_writter = log_to_file(new_log_path)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=5, repeat=1, min_repeat_ms=1000)
    )
    measure_batch = create_measure_batch(task, measure_option)
    results = measure_batch(inputs)
    log_writter(None, inputs, results)

    del measure_batch
    return

if not os.path.exists(platform2_dir):
    os.mkdir(platform2_dir)

for config_file in tqdm.tqdm(glob.glob('{}/*.json'.format(platform1_dir))):
    parse_one_log(config_file, platform2_dir)

for wkl, (new_log_path, heap) in tqdm.tqdm(wkls.items()):
    if not heap:
        continue
    run_one_wkl(wkl, new_log_path, [item[1] for item in heap])

