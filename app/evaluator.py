"""Customized AutoTVM Builder and Runner for Cost Models."""
# pylint: disable=ungrouped-imports

import time

import mxnet as mx
import numpy as np
import scipy.stats as ss
from mxnet import npx

from perf_model.data_proc import extract_feature
from tvm.autotvm.measure import LocalRunner, MeasureErrorNo, MeasureResult
from tvm.autotvm.measure.measure import Builder
from tvm.autotvm.measure.measure_methods import BuildResult

npx.set_np()

class DummyBuilder(Builder):
    """A dummy builder for cost model."""

    def build(self, measure_inputs):
        """Build nothing."""
        return [BuildResult(None, None, None, 0) for _ in range(len(measure_inputs))]


class RankModelRunner(LocalRunner):
    """Rank Model Runner Class."""
    def __init__(self,
                 valid_model_forward,
                 rank_model_forward,
                 feature_metadata,
                 verify_model_accuracy=False,
                 timeout=5,
                 number=4,
                 repeat=3,
                 min_repeat_ms=0,
                 cooldown_interval=0.1,
                 check_correctness=False):
        super(RankModelRunner, self).__init__(timeout, number, repeat, min_repeat_ms,
                                              cooldown_interval, check_correctness)
        self.valid_model_forward = valid_model_forward
        self.rank_model_forward = rank_model_forward
        self.feture_metadata = feature_metadata
        self.verify_model_accuracy = verify_model_accuracy
        self.model_accuracy = [0, 0, 0]  # (total, valid error, rank error)

    def get_model_acc(self):
        """Get the accuracy of valid and rank models."""
        if self.model_accuracy[0] == 0:
            return (0, 0)
        return (self.model_accuracy[1] / self.model_accuracy[0],
                self.model_accuracy[2] / self.model_accuracy[0])

    def set_task(self, task):
        self.task = task
        if self.verify_model_accuracy:
            return super(RankModelRunner, self).set_task(task)

    def get_build_kwargs(self):
        if self.verify_model_accuracy:
            return super(RankModelRunner, self).get_build_kwargs()
        return {}

    def run(self, measure_inputs, build_results):
        """The cost in MeasureResult is the ranking. 1 is the best."""
        # Extract features
        features = []
        used_features = []
        for inp in measure_inputs:
            feat_dict = extract_feature(inp)

            feat = []
            used_feat = []
            for name, used, meta in self.feture_metadata:
                val = None
                if name in feat_dict:
                    if isinstance(meta[0], str):
                        val = meta.index(feat_dict[name])
                    elif isinstance(meta[0], float):
                        val = (feat_dict[name] - meta[0]) / meta[1]
                feat.append(val)
                if used:
                    used_feat.append(val)
            features.append(feat)
            used_features.append(used_feat)

        nd_features = mx.np.array(features)
        nd_used_features = mx.np.array(used_features)

        # Run valid model.
        valids = self.valid_model_forward(nd_features)

        # Pairwise ranking.
        scores = np.zeros(len(nd_used_features), dtype='int32')
        for idx1, feat1 in enumerate(nd_used_features):
            if not valids[idx1]:
                scores[idx1] = -1 # Make sure invalid configs will have the lowest ranking.
                continue
            for idx2, feat2 in enumerate(nd_used_features[idx1 + 1:]):
                if not valids[idx2]:
                    continue
                rank = self.rank_model_forward(mx.np.array([feat1, feat2]))
                if rank[0] == 0:
                    scores[idx1] += 1
                elif rank[1] == 0:
                    scores[idx2] += 1
        ranks = (ss.rankdata(-scores, method='dense')).tolist()

        results = []
        for idx, valid in enumerate(valids):
            if not valid:
                results.append(MeasureResult([1e+5], MeasureErrorNo.NO_ERROR, 0, time.time()))
            else:
                results.append(MeasureResult([ranks[idx]], MeasureErrorNo.NO_ERROR, 0,
                                             time.time()))

        if self.verify_model_accuracy:
            real_results = super(RankModelRunner, self).run(measure_inputs, build_results)
            self.model_accuracy[0] += len(real_results)

            for valid, ret in zip(valids, real_results):
                real_valid = ret.error_no != MeasureErrorNo.NO_ERROR
                self.model_accuracy[1] += 1 if real_valid != valid else 0

            costs = [
                np.mean(r.costs) if v and r.error_no == MeasureErrorNo.NO_ERROR else 10e+5
                for v, r in zip(valids, real_results)
            ]
            real_ranks = ss.rankdata(costs, method='dense').tolist()

            for pred, real in zip(ranks, real_ranks):
                self.model_accuracy[2] += 1 if pred != real else 0

        return results
