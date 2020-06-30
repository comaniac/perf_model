"""A tuner to make use of ranking based cost model."""

from tvm.autotvm.tuner import RandomTuner


class RoundTuner(RandomTuner):
    """Tune the task with a latency ranking model."""
    def __init__(self, task, n_cfg):
        super(RoundTuner, self).__init__(task)

        # Maintain the top rank configs.
        self.n_cfg = n_cfg
        self.top_cfgs = []
        self.pivot_cfg = None

    def get_top_rank_cfgs(self):
        return self.top_cfgs

    def next_batch(self, batch_size):
        """Generate the next batch with the pivot"""
        if self.pivot_cfg is None:
            return super(RoundTuner, self).next_batch(batch_size)
        return super(RoundTuner,
                     self).next_batch(batch_size - 1) + [self.pivot_cfg]

    def update(self, inputs, results):
        """Update top rank configs and the pivot."""

        # Sort configs by results. result.costs should be the ranking instead of latency.
        ranked_cfgs = [
            x[0].config
            for x in sorted(zip(inputs, results), key=lambda x: x[1].costs[0])
        ]

        pivot_cfg_str = str(
            self.pivot_cfg) if self.pivot_cfg is not None else None
        if pivot_cfg_str is None:
            # Take n_cfgs from the first batch to make sure we have enough cfgs.
            self.top_cfgs = ranked_cfgs[:min(len(inputs), self.n_cfg)]
        elif pivot_cfg_str != str(ranked_cfgs[0]):
            # Take cfgs with higher ranks than the pivot.
            pivot_idx = -1
            for idx, cfg in enumerate(ranked_cfgs):
                if pivot_cfg_str == str(cfg):
                    pivot_idx = idx
                    break
            if pivot_idx == -1:
                raise RuntimeError('Cannot find the pivot in the batch')
            self.top_cfgs = ranked_cfgs[:pivot_idx] + self.top_cfgs
        else:
            # Pivot is still the best. Throw away an entire batch.
            pass

        # Update the pivot and maintain the size of top cfgs.
        self.pivot_cfg = ranked_cfgs[0]
        if len(self.top_cfgs) > self.n_cfg:
            self.top_cfgs = self.top_cfgs[:self.n_cfg]
