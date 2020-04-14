"""Run AutoTVM with the Ranking Model."""
# pylint: disable=ungrouped-imports

import glob
import os
import sys

import numpy as np
from mxnet import npx

import tvm
import tvm.contrib.graph_runtime as runtime
from evaluator import DummyBuilder, RankModel, RankModelRunner
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
    """Main entry.
    Usage: <model-dir>
    Model directory should be organized as:
    target-models/task-name/{valid_net.*, embed_net.*, rank_score_net.*, feature.meta}
    """

    # Map from task name to model.
    models = {}
    for model_path in glob.glob(sys.argv[1]):
        task_name = os.path.basename(model_path)
        models[task_name] = RankModel(task_name, model_path)

    verify_model = False
    measure_option = autotvm.measure_option(
        builder=DummyBuilder() if not verify_model else LocalBuilder(),
        runner=RankModelRunner(models=models,
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
