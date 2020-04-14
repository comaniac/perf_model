"""Run AutoTVM with the Ranking Model."""
# pylint: disable=ungrouped-imports

import sys
import warnings

import mxnet as mx
import numpy as np
from mxnet import gluon, npx

import tvm
import tvm.contrib.graph_runtime as runtime
from evaluator import DummyBuilder, RankModelRunner
from round_tuner import RoundTuner
from tvm import autotvm, relay
from tvm.autotvm.measure import MeasureInput, create_measure_batch
from tvm.autotvm.measure.measure_methods import LocalBuilder, LocalRunner
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner
from tvm.relay import testing

npx.set_np()


def get_network(name, dtype, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = testing.resnet.get_workload(num_layers=n_layer,
                                                  batch_size=batch_size,
                                                  dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = testing.vgg.get_workload(num_layers=n_layer,
                                               batch_size=batch_size,
                                               dtype=dtype)
    elif name == 'mobilenet':
        mod, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = testing.squeezenet.get_workload(batch_size=batch_size,
                                                      version='1.1',
                                                      dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
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

    if tuner == 'round':
        remeasure_option = autotvm.measure_option(
            builder=LocalBuilder(),
            runner=LocalRunner(number=5, repeat=3, min_repeat_ms=1000),
        )
    else:
        remeasure_option = None

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        n_trial = 8000  #len(task.config_space)

        callbacks = [
            autotvm.callback.progress_bar(n_trial, prefix=prefix),
            autotvm.callback.log_to_file(log_filename)
        ]

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        elif tuner == 'round':
            tuner_obj = RoundTuner(task, n_cfg=8)
            callbacks = callbacks[:1] # Do not record ranks.
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=callbacks)

        # Round tuner needs an extra measurement step to get the real throughputs.
        if tuner == 'round':
            top_cfgs = tuner_obj.get_top_rank_cfgs()
            measure_batch = create_measure_batch(task, remeasure_option)
            inputs = [MeasureInput(task.target, task, config) for config in top_cfgs]
            print(' Re-measuring %d top configs' % len(inputs), end='')
            results = measure_batch(inputs)
            print(', exporting', end='')
            autotvm.callback.log_to_file(log_filename)(None, inputs, results)


def tune_and_evaluate(model_name, dtype, batch_size, target, tuning_opt):
    """Tune a model with the ranking model and evaluate the performance."""

    print("Extract tasks (conv2d only for now)...")
    mod, params, input_shape, _ = get_network(model_name, dtype, batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"), ))
    #tasks = tasks[:1]

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

    # Load feature metadata.
    feature_metadata = []
    with open(sys.argv[1], 'r') as filep:
        for line in filep:
            tokens = line.replace('\n', '').split(',')
            try:
                # Numerical features: (name, used, (mean, std))
                vals = [float(t) for t in tokens[1:]]
                # Std = 1 means no effect because it was 0 and we workaround it to 1.
                feature_metadata.append((tokens[0], vals[1] != 1, vals))
            except ValueError:
                # Categorized features: (name, used, [options])
                feature_metadata.append((tokens[0], len(tokens) > 2, tokens[1:]))

    net_dir = sys.argv[2]
    valid_net_file = '{}/valid_net'.format(net_dir)
    embed_net_file = '{}/embed_net'.format(net_dir)
    rank_net_file = '{}/rank_score_net'.format(net_dir)

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
        preds = valid_net(features)
        return [(p[0] <= p[1]).tolist() for p in preds]

    def rank_model_forward(features):
        """Rank Model Inference."""
        assert len(features) == 2
        embeddings = embed_net(features)
        lhs_embeddings = mx.np.array([embeddings[0]])
        inner_lhs_embeddings = mx.np.expand_dims(lhs_embeddings, axis=1)

        rhs_embeddings = mx.np.array([embeddings[1]])
        rhs_embeddings = mx.np.expand_dims(rhs_embeddings, axis=0)

        inner_lhs_embeddings, rhs_embeddings = mx.np.broadcast_arrays(
            inner_lhs_embeddings, rhs_embeddings)

        joint_embedding = mx.np.concatenate([
            inner_lhs_embeddings, rhs_embeddings,
            mx.np.abs(inner_lhs_embeddings - rhs_embeddings), inner_lhs_embeddings * rhs_embeddings
        ],
                                            axis=-1)

        pred_rank_label_scores = rank_net(joint_embedding)
        pred = pred_rank_label_scores.argmax(axis=-1).astype(np.int32)
        if pred == 0:
            return [1, 1]
        elif pred == 1:
            return [0, 1]
        return [1, 0]

    verify_model = False
    measure_option = autotvm.measure_option(
        builder=DummyBuilder() if not verify_model else LocalBuilder(),
        runner=RankModelRunner(valid_model_forward=valid_model_forward,
                               rank_model_forward=rank_model_forward,
                               feature_metadata=feature_metadata,
                               verify_model_accuracy=verify_model,
                               number=5,
                               repeat=1,
                               min_repeat_ms=1000),
    )
    tuning_option = {
        'log_filename': 'tune.log',
        'tuner': 'round',
        'early_stopping': None,
        'measure_option': measure_option,
    }
    tune_and_evaluate('resnet-18', 'float32', 1, 'cuda --model=t4', tuning_option)

    if verify_model:
        valid, rank = measure_option['runner'].get_model_acc()
        print('\nModel accuracy: Valid %.2f%%; Rank %.2f%%' % (valid * 100.0, rank * 100.0))


if __name__ == "__main__":
    main()
