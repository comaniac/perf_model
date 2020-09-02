"""Customized AutoTVM Builder and Runner for Cost Models."""
# pylint: disable=ungrouped-imports

from concurrent.futures import ProcessPoolExecutor
import os
import sys
import time
import warnings

import mxnet as mx
import numpy as np
import scipy.stats as ss
from mxnet import gluon, npx
from sklearn.metrics import ndcg_score

try:
    import catboost
except Exception:  # pylint: disable=broad-except
    import imp
from perf_model.data_proc import extract_feature
from tvm.autotvm.measure import LocalRunner, MeasureErrorNo, MeasureResult
from tvm.autotvm.measure.measure import Builder
from tvm.autotvm.measure.measure_methods import BuildResult


class DummyBuilder(Builder):
    """A dummy builder for cost model."""
    def __init__(self, n_parallel=8):
        """We can set a large value of n_parallel since we do not really build configs."""
        super(DummyBuilder, self).__init__(n_parallel=n_parallel)

    def build(self, measure_inputs):
        """Build nothing."""
        return [
            BuildResult(None, None, None, 0)
            for _ in range(len(measure_inputs))
        ]


class RankModel():
    """A Ranking Model with A Valid Model."""
    def __init__(self, task_name, model_path):
        # Parse feature metadata.
        meta_file = os.path.join(model_path, 'feature.meta')
        if not os.path.exists(meta_file):
            raise RuntimeError('Feature metadata for %s is missing in %s' %
                               (task_name, model_path))
        self.feature_metadata = []
        with open(meta_file, 'r') as filep:
            for line in filep:
                tokens = line.replace('\n', '').split(',')
                if tokens[1] == 'category':
                    # Categorized features: (name, used, type, [options])
                    self.feature_metadata.append(
                        (tokens[0], len(tokens) > 3, tokens[1], tokens[2:]))
                else:
                    try:
                        # Numerical features: (name, used, type, (mean, std))
                        vals = [float(t) for t in tokens[2:]]
                        assert len(vals) == 2
                        # Std = 1 means no effect because it was 0 and we workaround it to 1.
                        self.feature_metadata.append(
                            (tokens[0], vals[1] != 1, tokens[1], vals))
                    except ValueError:
                        raise RuntimeError(
                            '%s is numeric but cannot be converted to float' %
                            tokens[0])

        self.load_models(model_path)

    def load_models(self, model_path):
        raise NotImplementedError

    def valid_model_forward(self, features):
        """Valid Model Inference."""
        raise NotImplementedError

    def rank_model_forward(self, valids, features):
        """Rank Model Inference."""
        raise NotImplementedError


class RankModelRunner(LocalRunner):
    """Rank Model Runner Class."""
    def __init__(self,
                 models,
                 verify_model_accuracy=False,
                 timeout=5,
                 number=4,
                 repeat=3,
                 enable_cpu_cache_flush=False,
                 min_repeat_ms=0,
                 cooldown_interval=0.1,
                 check_correctness=False):
        super(RankModelRunner,
              self).__init__(timeout, number, repeat, min_repeat_ms,
                             cooldown_interval, check_correctness)
        self.models = models
        self.verify_model_accuracy = verify_model_accuracy
        self.model_accuracy = [0, 0, 0]  # (total, valid error, rank error)
        self.local_runner = LocalRunner(number=number,
                                        repeat=repeat,
                                        min_repeat_ms=min_repeat_ms,
                                        enable_cpu_cache_flush=enable_cpu_cache_flush)

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
        return None, None

    def get_build_kwargs(self):
        if self.verify_model_accuracy:
            return super(RankModelRunner, self).get_build_kwargs()
        return {}

    def extract_features(self, measure_inputs):
        """Extract features from measure inputs."""
        task_name = ''
        features = []
        used_features = []
        for inp in measure_inputs:
            task_name = inp.task.name
            if task_name not in self.models:
                raise RuntimeError('No cost model for %s' % task_name)

            feat_dict = extract_feature(inp)

            feat = []
            used_feat = []
            for name, used, feat_type, meta in self.models[task_name].feature_metadata:
                val = None
                if name in feat_dict:
                    if feat_type == 'numeric':
                        val = (feat_dict[name] - meta[0]) / meta[1]
                    else:
                        val = meta.index(str(feat_dict[name]))
                else:
                    # It happens when this parameter is missing in this config.
                    # For example, "unroll_kw" only appears at the conv2d.x86
                    # with kernel size 1x1.
                    if feat_type == 'numeric':
                        val = 0
                    else:
                        if 'None' not in meta:
                            # Meaning that training data does not cover a config
                            # without this parameter. The prediction result
                            # may not be accurate.
                            val = 0
                        else:
                            val = meta.index('None')
                feat.append(val)
                if used:
                    used_feat.append(val)
            features.append(feat)
            used_features.append(used_feat)

        return task_name, features, used_features

    def run(self, measure_inputs, build_results):
        """The cost in MeasureResult is the ranking. Lower the better."""
        task_name, features, used_features = self.extract_features(measure_inputs)

        # Run valid model.
        valids = self.models[task_name].valid_model_forward(features)

        # Run ranking model.
        ranks = self.models[task_name].rank_model_forward(valids, used_features)

        # Create results.
        results = []
        for idx, valid in enumerate(valids):
            if not valid:
                results.append(
                    MeasureResult([1e+5], MeasureErrorNo.NO_ERROR, 0,
                                  time.time()))
            else:
                results.append(
                    MeasureResult([ranks[idx]], MeasureErrorNo.NO_ERROR, 0,
                                  time.time()))

        if self.verify_model_accuracy:
            real_results = super(RankModelRunner, self).run(measure_inputs, build_results)
            self.model_accuracy[0] += len(real_results)

            for valid, ret in zip(valids, real_results):
                real_valid = ret.error_no != MeasureErrorNo.NO_ERROR
                self.model_accuracy[1] += 1 if real_valid != valid else 0

            costs = [
                np.mean(r.costs)
                if v and r.error_no == MeasureErrorNo.NO_ERROR else 10e+5
                for v, r in zip(valids, real_results)
            ]
            ndcg = ndcg_score(y_true=[costs],
                              y_score=[[r.costs[0] for r in results]])
            self.model_accuracy[2] += ndcg

        return results


class ListwiseRankModel(RankModel):
    """A Elementwise Ranking Model with A Valid Model."""

    def load_models(self, model_path):
        """Load models."""
        valid_net_file = '{}/valid_model.cbm'.format(model_path)
        rank_net_cbm = '{}/list_rank_net.cbm'.format(model_path)
        rank_net_py = '{}/list_rank_net.py'.format(model_path)
        self.path = model_path
        self.is_cbm_model = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if os.path.exists(valid_net_file):
                self.valid_net = catboost.CatBoostClassifier().load_model(valid_net_file)
            else:
                self.valid_net = None

            try:
                self.rank_net = catboost.CatBoost().load_model(rank_net_cbm)
            except NameError: # CatBoost is unavailable. Try to load Python model.
                self.is_cbm_model = False
                with open(rank_net_py, 'rb') as filep:
                    self.rank_net = imp.load_module(
                        model_path.replace('/', '_').replace('.', '_'), filep,
                        'list_rank_net.py', ('.py', 'rb', imp.PY_SOURCE))

    def valid_model_forward(self, features):
        """Valid Model Inference."""
        if self.valid_net is not None:
            valids = self.valid_net.predict(features)
            return [v == 1 for v in valids]
        return [True] * len(features)

    def rank_model_forward(self, valids, features):
        """Rank Model Inference."""
        # Larger score is better.
        try:
            if self.is_cbm_model:
                scores = self.rank_net.predict(features)
            else:
                with ProcessPoolExecutor(max_workers=8) as pool:
                    scores = list(pool.map(self.rank_net.apply_catboost_model, features))
        except Exception as err:  # pylint: disable=broad-except
            sys.stderr.write(str(err))
            sys.stderr.write('Error at %s' % self.path)
            assert False


        # Smaller rank score is better.
        return [-s if v else 1e+5 for v, s in zip(valids, scores)]


def rank_progress(total, prefix):
    """Display progress bar for ranking process.

    Parameters
    ----------
    total: int
        The total number of trials.

    prefix: str
        The prefix string for the progress bar.
    """
    class _Context():
        """Context to store local variables"""
        def __init__(self):
            self.curr_cnt = 0
            self.total = total

    ctx = _Context()
    tic = time.time()

    sys.stdout.write('\r%s Progress: (%d/%d) | %.2f s' %
                     (prefix, 0, total, time.time() - tic))
    sys.stdout.flush()

    def _callback(tuner, inputs, results):  # pylint: disable=unused-argument
        ctx.curr_cnt += len(inputs)
        if ctx.curr_cnt >= ctx.total:
            sys.stdout.write('\r')
        else:
            sys.stdout.write(
                '\r%s Progress: (%d/%d) | %.2f s' %
                (prefix, ctx.curr_cnt, ctx.total, time.time() - tic))
        sys.stdout.flush()

    return _callback
