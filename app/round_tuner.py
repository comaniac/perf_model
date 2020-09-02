"""A tuner to make use of ranking based cost model."""

import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tvm.autotvm.graph_tuner.base_graph_tuner import get_infer_layout
from tvm.autotvm.measure import MeasureInput, MeasureResult
from tvm.autotvm.tuner import RandomTuner


class BestRecordByLayout:
    """A helper class to maintain the best record of each data layout."""
    @staticmethod
    def _infer_task_layout(
            record: Tuple[MeasureInput, MeasureResult]) -> Optional[Tuple]:
        """Infer the layout of the given task. Return None if the layout cannot be inferred.

        Parameters
        ----------
        record: Tuple[MeasureInput, MeasureResult]
            The AutoTVM record pair.

        Return
        ------
        layout: Optional[Tuple]
            A tuple of input and output layout, or None if not inferrable.
        """
        infer_layout_func: Optional[Callable] = None
        try:
            infer_layout_func = get_infer_layout(record[0].task.name)
            assert infer_layout_func is not None
            with record[0].target:
                return infer_layout_func(record[0].task.workload,
                                         record[0].config)
        except ValueError:
            pass
        return None

    def __init__(self, max_cfg_per_layout: int = 0):
        # Map from layout to a min-heap of records.
        self._data: Dict[Any, List[Any]] = {}
        self.max_cfg_per_layout = max_cfg_per_layout if max_cfg_per_layout > 0 else float('inf')
        self.task_str: Optional[str] = None

    def push(self, record: Tuple[MeasureInput, MeasureResult]):
        """Push a new record to the bucket based on its layout.

        Parameters
        ----------
        record: Tuple[MeasureInput, MeasureResult]
            The record to be pushed.
        """
        if self.task_str is None:
            self.task_str = str(record[0].task)
        assert self.task_str == str(record[0].task), 'Only accept the records of the same task'

        layout = self._infer_task_layout(record)
        if layout not in self._data:
            self._data[layout] = []

        # The min-heap key is composed of (-latency, timestamp)
        heapq.heappush(
            self._data[layout],
            ((-np.mean(record[1].costs), record[1].timestamp), record))
        if len(self._data[layout]) > self.max_cfg_per_layout:
            heapq.heappop(self._data[layout])

    def to_list(self, max_n_layout=1) -> List[Tuple[MeasureInput, MeasureResult]]:
        """Sort the record (of any layout) to be a list."""

        # Sort layouts by the best config.
        sorted_layouts = sorted(self._data.values(),
                                key=lambda h: max(h, key=lambda p: -p[0][0]),
                                reverse=True)
        if len(sorted_layouts) > max_n_layout:
            sorted_layouts = sorted_layouts[:max_n_layout]

        records: List[Tuple[MeasureInput, MeasureResult]] = []
        for heap in sorted_layouts:
            records += [p[1] for p in heap]
        return records


class RoundTuner(RandomTuner):
    """Tune the task with a latency ranking model."""
    def __init__(self, task, n_cfg):
        super(RoundTuner, self).__init__(task)
        self.best_cfgs_by_layout = BestRecordByLayout(n_cfg)

    def get_top_rank_cfgs(self, n_layout):
        return [inp.config for inp, _ in self.best_cfgs_by_layout.to_list(n_layout)]

    def update(self, inputs, results):
        """Update top configs."""
        for record in zip(inputs, results):
            self.best_cfgs_by_layout.push(record)
