"""Run AutoTVM with the Ranking Model."""
# pylint: disable=ungrouped-imports

import sys
import time
import warnings

import mxnet as mx
import numpy as np
from mxnet import gluon

import tvm
import tvm.contrib.graph_runtime as runtime
from data_proc import extract_feature
from tvm import autotvm, relay
from tvm.autotvm.measure import LocalRunner, MeasureErrorNo, MeasureResult
from tvm.autotvm.measure.measure import Builder
from tvm.autotvm.measure.measure_methods import BuildResult
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner

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
                 timeout=5,
                 n_parallel=None,
                 number=4,
                 repeat=3,
                 min_repeat_ms=0,
                 cooldown_interval=0.1,
                 check_correctness=False):
        super(RankModelRunner, self).__init__(timeout, n_parallel, number, repeat, min_repeat_ms,
                                              cooldown_interval, check_correctness)
        self.valid_model_forward = valid_model_forward
        self.rank_model_forward = rank_model_forward

    def set_task(self, task):
        self.task = task

    def run(self, measure_inputs, _):
        """The cost in MeasureResult is the ranking. 1 is the best."""
        features = extract_feature(measure_inputs)
        valids = self.valid_model_forward(features)

        # Pairwise ranking.
        scores = np.zeros(len(features))
        for idx1, feat1 in enumerate(features):
            if not valids[idx1]:
                continue
            for idx2, feat2 in enumerate(features[idx1 + 1:]):
                if not valids[idx2]:
                    continue
                rank = self.rank_model_forward([feat1, feat2])
                if rank[0] == 0:
                    scores[idx1] += 1
                elif rank[1] == 0:
                    scores[idx2] += 1
        ranks = np.argsort(-scores) + 1

        results = []
        for idx, valid in enumerate(valids):
            if not valid:
                results.append(MeasureResult([1e+5], MeasureErrorNo.RUNTIME_DEVICE, 0, time.time()))
            else:
                results.append(MeasureResult([ranks[idx]], MeasureErrorNo.NO_ERROR, 0, time.time()))
        return results

def get_network(name, dtype, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer,
                                                        batch_size=batch_size,
                                                        dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer,
                                                     batch_size=batch_size,
                                                     dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size,
                                                            version='1.1',
                                                            dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params,
                             net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


def tune_kernels(tasks,
                 measure_option,
                 tuner='random',
                 early_stopping=None,
                 log_filename='tuning.log'):
    """Tune kernels with the ranking model."""

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)
                       ])


def tune_and_evaluate(model_name, dtype, batch_size, target, tuning_opt):
    """Tune a model with the ranking model and evaluate the performance."""

    print("Extract tasks (conv2d only for now)...")
    mod, params, input_shape, _ = get_network(model_name, dtype, batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"), ))
    tasks = tasks[:1]

    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)

    with autotvm.apply_history_best(tuning_opt['log_filename']):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


def main():
    """Main entry."""

    valid_net_file = sys.argv[1]
    embed_net_file = sys.argv[2]
    rank_net_file = sys.argv[3]

    # Load models.
    ctx = mx.cpu(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid_net = gluon.nn.SymbolBlock.imports("{}-symbol.json".format(valid_net_file), ['data'],
                                                 "{}-0000.params".format(valid_net_file),
                                                 ctx=ctx)
        embed_net = gluon.nn.SymbolBlock.imports("{}-symbol.json".format(embed_net_file), ['data'],
                                                 "{}-0000.params".format(embed_net_file),
                                                 ctx=ctx)
        rank_net = gluon.nn.SymbolBlock.imports("{}-symbol.json".format(rank_net_file), ['data'],
                                                "{}-0000.params".format(rank_net_file),
                                                ctx=ctx)

    def valid_model_forward(features):
        """Valid Model Inference."""
        pred = valid_net(features)
        return pred[0] > pred[1]

    def rank_model_forward(features):
        """Rank Model Inference."""
        assert len(features) == 2
        embeddings = embed_net(features)
        lhs_embeddings = [embeddings[0]]
        inner_lhs_embeddings = mx.np.expand_dims(lhs_embeddings, axis=1)

        rhs_embeddings = [embeddings[1]]
        rhs_embeddings = mx.np.expand_dims(rhs_embeddings, axis=0)

        inner_lhs_embeddings, rhs_embeddings = mx.np.broadcast_arrays(
            inner_lhs_embeddings, rhs_embeddings)

        joint_embedding = mx.np.concatenate([inner_lhs_embeddings, rhs_embeddings,
                                             mx.np.abs(inner_lhs_embeddings - rhs_embeddings),
                                             inner_lhs_embeddings * rhs_embeddings], axis=-1)

        pred_rank_label_scores = rank_net(joint_embedding)
        pred = pred_rank_label_scores.argmax(axis=-1).astype(np.int32)
        if pred == 0:
            return [1, 1]
        elif pred == 1:
            return [0, 1]
        return [1, 0]


    measure_option = autotvm.measure_option(
        builder=DummyBuilder(),
        runner=RankModelRunner(valid_model_forward=valid_model_forward,
                               rank_model_forward=rank_model_forward,
                               number=10,
                               repeat=1,
                               min_repeat_ms=1000),
    )
    tuning_option = {
        'log_filename': 'tune.log',
        'tuner': 'random',  # TODO: tournament
        'early_stopping': None,
        'measure_option': measure_option,
    }
    tune_and_evaluate('resnet-18', 'float32', 1, 'cuda --model=t4', tuning_option)


if __name__ == "__main__":
    main()
