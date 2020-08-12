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

    def __init__(self, max_cfg_per_layout: int = 0, max_n_layout: int = 1):
        # Map from layout to a min-heap of records.
        self._data: Dict[Any, List[Any]] = {}
        self.max_cfg_per_layout = max_cfg_per_layout if max_cfg_per_layout > 0 else float('inf')
        self.max_n_layout = max_n_layout
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

        layout = None
        if self.max_n_layout > 1:
            layout = self._infer_task_layout(record)

        if layout not in self._data:
            self._data[layout] = []

        # The min-heap key is composed of (-latency, timestamp)
        heapq.heappush(
            self._data[layout],
            ((-np.mean(record[1].costs), record[1].timestamp), record))
        if len(self._data[layout]) > self.max_cfg_per_layout:
            heapq.heappop(self._data[layout])

    def to_list(self) -> List[Tuple[MeasureInput, MeasureResult]]:
        """Sort the record (of any layout) to be a list."""

        # Sort layouts by the best config.
        sorted_layouts = sorted(self._data.values(), key=lambda h: max(h, key=lambda p: -p[0][0]), reverse=True)
        if len(sorted_layouts) > self.max_n_layout:
            sorted_layouts = sorted_layouts[:self.max_n_layout]

        records: List[Tuple[MeasureInput, MeasureResult]] = []
        for heap in sorted_layouts:
            records += [p[1] for p in heap]
        return records


class RoundTuner(RandomTuner):
    """Tune the task with a latency ranking model."""
    def __init__(self, task, n_cfg, n_layout=1):
        super(RoundTuner, self).__init__(task)
        self.best_cfgs_by_layout = BestRecordByLayout(n_cfg, n_layout)

    def get_top_rank_cfgs(self):
        return [inp.config for inp, _ in self.best_cfgs_by_layout.to_list()]

    def update(self, inputs, results):
        """Update top configs."""
        for record in zip(inputs, results):
            self.best_cfgs_by_layout.push(record)

    # [Deprecate] The next_batch function for pairwise ranking model.
    # def next_batch(self, batch_size):
    #     """Generate the next batch with the pivot"""
    #     if self.pivot_cfg is None:
    #         return super(RoundTuner, self).next_batch(batch_size)
    #     return super(RoundTuner, self).next_batch(batch_size - 1) + [self.pivot_cfg]

    # def update(self, inputs, results):
    #     """Update top rank configs and the pivot."""

    #     # Sort configs by results. result.costs should be the ranking instead of latency.
    #     ranked_cfgs = [
    #         x[0].config
    #         for x in sorted(zip(inputs, results), key=lambda x: x[1].costs[0])
    #     ]

    #     pivot_cfg_str = str(
    #         self.pivot_cfg) if self.pivot_cfg is not None else None
    #     if pivot_cfg_str is None:
    #         # Take n_cfgs from the first batch to make sure we have enough cfgs.
    #         self.top_cfgs = ranked_cfgs[:min(len(inputs), self.n_cfg)]
    #     elif pivot_cfg_str != str(ranked_cfgs[0]):
    #         # Take cfgs with higher ranks than the pivot.
    #         pivot_idx = -1
    #         for idx, cfg in enumerate(ranked_cfgs):
    #             if pivot_cfg_str == str(cfg):
    #                 pivot_idx = idx
    #                 break
    #         if pivot_idx == -1:
    #             raise RuntimeError('Cannot find the pivot in the batch')
    #         self.top_cfgs = ranked_cfgs[:pivot_idx] + self.top_cfgs
    #     else:
    #         # Pivot is still the best. Throw away an entire batch.
    #         pass

    #     # Update the pivot and maintain the size of top cfgs.
    #     self.pivot_cfg = ranked_cfgs[0]
    #     if len(self.top_cfgs) > self.n_cfg:
    #         self.top_cfgs = self.top_cfgs[:self.n_cfg]
