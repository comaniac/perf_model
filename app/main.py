"""Run AutoTVM with the Ranking Model."""
# pylint: disable=ungrouped-imports

import argparse
import glob
import os
import re
import sys
import time
import logging
import random
import numpy as np

import tvm
import tvm.contrib.graph_runtime as runtime
from evaluator import DummyBuilder, ListwiseRankModel, NNRankModel, CatRegressionModel,\
    RankModelRunner, rank_progress, CatRankingModel
from round_tuner import RoundTuner
from tvm import autotvm, relay
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm.autotvm.measure import MeasureInput, create_measure_batch
from tvm.autotvm.measure.measure_methods import LocalBuilder
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner
from tvm.relay import testing
from tvm.relay.backend import compile_engine


def create_config():
    """Create the config parser of this app."""
    parser = argparse.ArgumentParser(description='Tune Model with Cost Model')
    parser.add_argument(
        '--list-net',
        default=None,
        help='The directory of trained listwise ranking models.'
        'Model directory should be organized as: '
        'target-models/task-name/{valid_net.*, list_rank_net.cbm, feature.meta}'
    )
    parser.add_argument('--model_type', default='cat_ranking',
                        choices=['nn',
                                 'cat_regression',
                                 'cat_ranking',
                                 'cat_ranking_old'])
    parser.add_argument('--target', required=True, help='The target platform')
    parser.add_argument('--n-parallel',
                        type=int,
                        default=8,
                        help='The batch size for config evaluation')
    parser.add_argument('--measure-top-n',
                        type=int,
                        default=32,
                        help='Number of top configs to be measured')
    parser.add_argument('--seed',
                        type=int,
                        default=123)
    parser.add_argument('--n-trial',
                        type=int,
                        default=5000,
                        help='The number of trials')
    parser.add_argument('--graph',
                        action='store_true',
                        help='Enable graph tuning (X86 only)')

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--gcv', help='model name in gluon cv model zoo')
    model_group.add_argument('--tf', help='TensorFlow model')
    model_group.add_argument('--pt', help='PyTorch model')
    model_group.add_argument('--test', help='Model name in Relay testing')

    return parser.parse_args()


def get_relay_test_model(name):
    """Get the symbol definition and random weight of a network"""

    dtype = 'float32'
    batch = 1
    input_shape = (batch, 3, 224,
                   224) if name.find('inception') == -1 else (batch, 3, 299,
                                                              299)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = testing.resnet.get_workload(num_layers=n_layer,
                                                  batch_size=batch,
                                                  dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = testing.vgg.get_workload(num_layers=n_layer,
                                               batch_size=batch,
                                               dtype=dtype)
    elif name == 'mobilenet':
        mod, params = testing.mobilenet.get_workload(batch_size=batch,
                                                     dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = testing.squeezenet.get_workload(batch_size=batch,
                                                      version='1.1',
                                                      dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = testing.inception_v3.get_workload(batch_size=batch,
                                                        dtype=dtype)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, ('data', input_shape)


def get_gcv_model(model_name):
    """Pull a Gluon CV model."""
    import gluoncv as gcv

    model_name = model_name.lower()

    shape = (1, 3, 224, 224)
    if model_name.find('inception') != -1:
        shape = (1, 3, 299, 299)
    elif model_name.find('yolo3') != -1:
        shape = (1, 3, 320, 320)
    elif model_name.startswith('ssd'):
        tokens = re.search(r'ssd_(\d+)_', model_name)
        size = int(tokens.group(1))
        shape = (1, 3, size, size)
    net = gcv.model_zoo.get_model(model_name, pretrained=True)
    mod, params = relay.frontend.from_mxnet(net, shape={'data': shape})
    return mod, params, ('data', shape)


def get_tf_model(model_path):
    """Load a TF model from a file."""
    from tvm.relay.frontend.tensorflow_parser import TFParser

    print('Loading TF model from file...')
    model_path, name, image_size = model_path.split(':')
    shape = (1, int(image_size), int(image_size), 3)
    logging.basicConfig(level=logging.CRITICAL)
    net = TFParser(model_path).parse()
    ret = relay.frontend.from_tensorflow(net,
                                         layout='NCHW',
                                         shape={name: shape})
    logging.basicConfig(level=logging.WARNING)
    return ret[0], ret[1], shape


def get_pt_model(model_name):
    """Pull a PyTorch model from Torch Vision."""
    import torch
    import torchvision

    print('Pull the model from Torch Vision...')
    shape = (1, 3, 224, 224)
    if model_name.find('inception') != -1:
        shape = (1, 3, 299, 299)

    model = getattr(torchvision.models, model_name)(pretrained=True)
    logging.basicConfig(level=logging.CRITICAL)
    model = model.eval()

    trace = torch.jit.trace(model, torch.randn(shape)).float().eval()
    logging.basicConfig(level=logging.WARNING)
    ret = relay.frontend.from_pytorch(trace, [('img', shape)])
    return ret[0], ret[1], ('img', shape)


def tune_kernels(tasks,
                 gen_graph_tuner_candidates,
                 measure_top_n,
                 measure_option,
                 tuner='random',
                 early_stopping=None,
                 n_trial=5000,
                 log_filename='tuning.log'):
    """Tune kernels with the ranking model."""

    remeasure_option = None
    if tuner == 'round':
        # Setup another measure option for final remeasurment.
        remeasure_option = autotvm.measure_option(
            builder=LocalBuilder(),
            runner=measure_option['runner'].local_runner,
        )
        assert isinstance(measure_option['runner'], RankModelRunner)

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        callbacks = []
        if task.name in ['dense_small_batch.cuda', 'conv2d_cudnn.cuda',
                         'dense_cublas.cuda', 'dense_large_batch.cuda']:
            # Ignore these four tasks
            continue
        if task.name not in measure_option['runner'].models:
            print('not covered by cost models')
            continue

        # create tuner
        if tuner == 'round':
            tuner_obj = RoundTuner(task, n_cfg=measure_top_n)
            callbacks = [rank_progress(n_trial, prefix=prefix)]  # Use different callbacks.
        else:
            if tuner in ('xgb', 'xgb-rank'):
                tuner_obj = XGBTuner(task, loss_type='rank')
            elif tuner == 'ga':
                tuner_obj = GATuner(task, pop_size=50)
            elif tuner == 'random':
                tuner_obj = RandomTuner(task)
            elif tuner == 'gridsearch':
                tuner_obj = GridSearchTuner(task)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            callbacks = [
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename)
            ]

        tic = time.time()

        # do tuning
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=callbacks)

        # Round tuner needs an extra measurement step to get the real throughputs.
        if tuner == 'round':
            max_n_layout = 20 if gen_graph_tuner_candidates else 1
            top_cfgs = tuner_obj.get_top_rank_cfgs(max_n_layout)
            measure_batch = create_measure_batch(task, remeasure_option)
            inputs = [
                MeasureInput(task.target, task, config) for config in top_cfgs
            ]
            sys.stderr.write('{} Measure Top {} Configs'.format(prefix, len(inputs)))
            results = measure_batch(inputs)

            best_idx, best_flops = max([
                (idx, i.task.flop / np.mean(r.costs) / 1e9 if r.error_no == 0 else 0)
                for idx, (i, r) in enumerate(zip(inputs, results))
            ], key=lambda x: x[1])

            sys.stderr.write(' | Best %.2f GFLOPS at Top %d | %.2fs\n' %
                             (best_flops, best_idx, time.time() - tic))
            autotvm.callback.log_to_file(log_filename)(None, inputs, results)



def tune_and_evaluate(mod, params, input_shape, dtype, measure_top_n, target, tuning_opt,
                      graph_log_file):
    """Tune a model with the ranking model and evaluate the performance."""

    sys.stderr.write("Extract conv2d tasks...\n")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # Run tuning tasks.
    if graph_log_file is not None and not os.path.exists(graph_log_file):
        tune_kernels(tasks, True, measure_top_n, **tuning_opt)
        tune_graph(mod["main"], input_shape[1], target, tuning_opt['log_filename'], graph_log_file)
    else:
        tune_kernels(tasks, False, measure_top_n, **tuning_opt)

    dispatch_ctx = tvm.autotvm.task.DispatchContext.current

    if graph_log_file is not None and os.path.exists(graph_log_file):
        sys.stderr.write("Compile model with graph tuning...\n")
        tvm.autotvm.task.DispatchContext.current = autotvm.apply_graph_best(graph_log_file)
    elif os.path.exists(tuning_opt['log_filename']):
        sys.stderr.write("Compile model without graph tuning...\n")
        tvm.autotvm.task.DispatchContext.current = autotvm.apply_history_best(
            tuning_opt['log_filename'])
    else:
        sys.stderr.write("Compile model with fallback + tophub...\n")

    compile_engine.get().clear()
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    tvm.autotvm.task.DispatchContext.current = dispatch_ctx

    # Load parameters.
    ctx = tvm.context(str(target), 0)
    module = runtime.create(graph, lib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape[1])).astype(dtype))
    module.set_input(input_shape[0], data_tvm)
    module.set_input(**params)

    # Evaluate performance.
    sys.stderr.write("Evaluate inference time cost...\n")
    ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    sys.stderr.write("Median inference time: %.2f ms\n" % np.median(prof_res))


def tune_graph(graph, dshape, target, records, opt_sch_file, use_dp=True):
    """Use graph tuner to minimize layout transform on CPU."""
    target_op = [relay.op.get("nn.conv2d"),]
    Tuner = DPTuner if use_dp else PBQPTuner
    executor = Tuner(graph, {'data': dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


def main():
    """Main entry."""
    configs = create_config()
    np.random.seed(configs.seed)
    random.seed(configs.seed)

    # Check if the target is for x86.
    target = tvm.target.create(configs.target)
    is_x86 = target.kind.name == 'llvm' and target.keys[0] == 'cpu'
    if not is_x86:
        measure_option = {
            'number': 5,
            'repeat': 1,
            'min_repeat_ms': 1000,
            'enable_cpu_cache_flush': False
        }
    else:
        measure_option = {
            'number': 1,
            'repeat': 10,
            'min_repeat_ms': 1,
            'enable_cpu_cache_flush': True
        }

    # Get the model.
    if configs.gcv:
        mod, params, input_shape = get_gcv_model(configs.gcv)
    elif configs.tf:
        mod, params, input_shape = get_tf_model(configs.tf)
    elif configs.pt:
        mod, params, input_shape = get_pt_model(configs.pt)
    else:
        mod, params, input_shape = get_relay_test_model(configs.test)

    # Map from task name to model.
    verify_model = False
    models = {}
    if configs.list_net is not None:
        for model_path in glob.glob('{}/*'.format(configs.list_net)):
            if not os.path.isdir(model_path):
                continue
            task_name = os.path.basename(model_path)
            if configs.model_type == 'cat_ranking_old':
                models[task_name] = ListwiseRankModel(task_name, model_path)
            elif configs.model_type == 'cat_ranking':
                models[task_name] = CatRankingModel(task_name, model_path)
            elif configs.model_type == 'cat_regression':
                models[task_name] = CatRegressionModel(task_name, model_path)
            elif configs.model_type == 'nn':
                models[task_name] = NNRankModel(task_name, model_path)
            else:
                raise NotImplementedError
            print('Loaded cost model for %s' % task_name)

    measure_option = autotvm.measure_option(
        builder=DummyBuilder(configs.n_parallel)
        if not verify_model else LocalBuilder(n_parallel=configs.n_parallel),
        runner=RankModelRunner(models=models,
                               verify_model_accuracy=verify_model,
                               **measure_option))
    tuning_option = {
        'log_filename': 'tune.log',
        'tuner': 'round',
        'early_stopping': None,
        'n_trial': configs.n_trial,
        'measure_option': measure_option,
    }

    if configs.graph and not is_x86:
        sys.stderr.write('WARNING: Graph tuner supports X86 only\n')
        configs.graph = False

    graph_log_file = 'graph.log' if configs.graph else None
    tune_and_evaluate(mod, params, input_shape, 'float32',
                      configs.measure_top_n, configs.target, tuning_option,
                      graph_log_file)

    if verify_model:
        valid, rank = measure_option['runner'].get_model_acc()
        sys.stderr.write('\nModel accuracy: Valid %.2f%%; NDCG %.3f\n' % (valid * 100.0, rank))


if __name__ == "__main__":
    main()
