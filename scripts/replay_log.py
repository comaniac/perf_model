"""Replay all configs in a given log file."""
# pylint: disable=unused-import
import sys

from tqdm import tqdm

import topi
import tvm
from tvm import autotvm
from tvm.autotvm.env import GLOBAL_SCOPE
from tvm.autotvm.measure import MeasureInput, create_measure_batch
from tvm.autotvm.measure.measure_methods import LocalBuilder, LocalRunner
from tvm.autotvm.record import load_from_file
from tvm.autotvm.tuner.callback import log_to_file


def batch_loader(log_file, target, batch_size=8):
    """Batch loading measure inputs."""
    tvm_target = tvm.target.create(target)
    batch = []
    for inp, _ in load_from_file(log_file):
        # FIXME (comaniac): If we apply different target (e.g., llvm to cuda) then
        # the task might be missing.
        inp.task.target = tvm_target
        new_inp = MeasureInput(tvm_target, inp.task, inp.config)
        batch.append(new_inp)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


def main():
    """Main function."""
    log_file = sys.argv[1]
    target = sys.argv[2]
    new_log_file = 'replay_{0}'.format(log_file)
    log_callback = log_to_file(new_log_file)

    GLOBAL_SCOPE.in_tuning = True
    measure_option = autotvm.measure_option(
        builder=LocalBuilder(),
        runner=LocalRunner(timeout=10, number=5, repeat=3, min_repeat_ms=1000)
    )
    measure_batch = None

    for batch in tqdm(batch_loader(log_file, target)):
        if measure_batch is None:
            measure_batch = create_measure_batch(batch[0].task, measure_option)

        results = measure_batch(batch)
        log_callback(None, batch, results)

    GLOBAL_SCOPE.in_tuning = False
    if measure_batch is not None:
        del measure_batch


if __name__ == '__main__':
    main()
