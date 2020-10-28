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
from evaluator import (
    DummyBuilder,
    ListwiseRankModel,
    NNRankModel,
    CatRegressionModel,
    RankModelRunner,
    rank_progress,
    CatRankingModel,
)
from round_tuner import RoundTuner
from tvm import autotvm, relay
from tvm.autotvm.measure import MeasureInput, create_measure_batch
from tvm.autotvm.measure.measure_methods import LocalBuilder
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner
from tvm.relay import testing
from tvm.relay.backend import compile_engine

model_shape_dict = {
    "reko/face_attribute/attributes": [(1, 3, 112, 112)],
    "reko/face_detection/face_v6_det_bifpn32_720_1280": [(1, 3, 720, 1280)],
    "reko/face_detection/face_v6_det_bifpn32_1024_1024": [(1, 3, 1024, 1024)],
    "reko/face_detection/face_v6_det_bifpn32_1080_1920": [(1, 3, 1080, 1920)],
    "reko/face_embedding/resnet_v5": [(1, 3, 112, 112)],
    "reko/face_tracking/detection-head-720p": [
        (1, 64, 182, 322),
        (1, 64, 92, 162),
        (1, 64, 47, 82),
        (1, 64, 25, 42),
        (1, 64, 14, 22),
        (1, 64, 8, 12),
        (1, 64, 182, 322),
        (1, 64, 92, 162),
        (1, 64, 47, 82),
        (1, 64, 25, 42),
        (1, 64, 14, 22),
        (1, 64, 8, 12),
        (200, 10),
        (200, 10),
    ],
    "reko/face_tracking/feature-backbone-720p": [(1, 3, 720, 1280)],
    "reko/face_tracking/motion-head-720p": [
        (1, 64, 182, 322),
        (1, 64, 92, 162),
        (1, 64, 47, 82),
        (1, 64, 25, 42),
        (1, 64, 14, 22),
        (1, 64, 8, 12),
        (1, 64, 182, 322),
        (1, 64, 92, 162),
        (1, 64, 47, 82),
        (1, 64, 25, 42),
        (1, 64, 14, 22),
        (1, 64, 8, 12),
    ],
}


def create_config():
    """Create the config parser of this app."""
    parser = argparse.ArgumentParser(description="Tune Model with Cost Model")
    parser.add_argument(
        "--list-net",
        default=None,
        help="The directory of trained listwise ranking models."
        "Model directory should be organized as: "
        "target-models/task-name/{valid_net.*, list_rank_net.cbm, feature.meta}",
    )
    parser.add_argument("--target", required=True, help="The target platform")
    parser.add_argument(
        "--n-parallel", type=int, default=8, help="The batch size for config evaluation"
    )
    parser.add_argument(
        "--measure-top-n", type=int, default=32, help="Number of top configs to be measured"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-trial", type=int, default=5000, help="The number of trials")

    parser.add_argument("--mx", help="MXNet model")

    return parser.parse_args()


def get_mxnet_network(model_name, dshape, dtype="float32"):
    """Parse MNXet graph.
    Args:
        model_name(str): the network artifact directory.
        dshape(list): the list of shapes for the input data.
        dtype(str): the data type of the input data.
        flush_denormal(bool): the flag to control whether flush the denormal number or not.
    Returns:
        tvm module: the tvm network.
        tvm parameter dictionary: the corresponding parameter dictionary of the network.
    """
    import mxnet as mx

    sm = mx.sym.load("{}-symbol.json".format(model_name))
    if len(dshape) == 1:
        mx_input = mx.sym.var("data")
        tvm_input = {"data": dshape[0]}
    else:
        mx_input, tvm_input = [], {}
        for i in range(len(dshape)):
            tvm_input["data{}".format(i)] = dshape[i]
            mx_input.append(mx.sym.var("data{}".format(i)))
    sym = mx.gluon.SymbolBlock(sm, mx_input)
    sym.hybridize()
    sym.collect_params().load("{}-0000.params".format(model_name))
    mod, params = relay.frontend.from_mxnet(sym, tvm_input, dtype=dtype)
    net = mod["main"]
    net = relay.Function(net.params, net.body, None, net.type_params, net.attrs)
    mod = tvm.IRModule.from_expr(net)
    return mod, params, tvm_input


def tune_kernels(
    tasks,
    measure_top_n,
    measure_option,
    tuner="random",
    early_stopping=None,
    n_trial=5000,
    log_filename="tuning.log",
):
    """Tune kernels with the ranking model."""

    remeasure_option = None
    if tuner == "round":
        # Setup another measure option for final remeasurment.
        remeasure_option = autotvm.measure_option(
            builder=LocalBuilder(),
            runner=measure_option["runner"].local_runner,
        )
        assert isinstance(measure_option["runner"], RankModelRunner)

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        callbacks = []
        if task.name not in measure_option["runner"].models:
            print("%s %s not covered by cost models" % (prefix, task.name))
            continue

        # create tuner
        if tuner == "round":
            tuner_obj = RoundTuner(task, n_cfg=measure_top_n)
            callbacks = [rank_progress(n_trial, prefix=prefix)]  # Use different callbacks.
        else:
            if tuner in ("xgb", "xgb-rank"):
                tuner_obj = XGBTuner(task, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(task, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(task)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(task)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            callbacks = [
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ]

        tic = time.time()

        # do tuning
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=callbacks,
        )

        # Round tuner needs an extra measurement step to get the real throughputs.
        if tuner == "round":
            top_cfgs = tuner_obj.get_top_rank_cfgs(1)
            measure_batch = create_measure_batch(task, remeasure_option)
            inputs = [MeasureInput(task.target, task, config) for config in top_cfgs]
            sys.stderr.write("{} Measure Top {} Configs".format(prefix, len(inputs)))
            results = measure_batch(inputs)

            best_idx, best_flops = max(
                [
                    (idx, i.task.flop / np.mean(r.costs) / 1e9 if r.error_no == 0 else 0)
                    for idx, (i, r) in enumerate(zip(inputs, results))
                ],
                key=lambda x: x[1],
            )

            sys.stderr.write(
                " | Best %.2f GFLOPS at Top %d | %.2fs\n"
                % (best_flops, best_idx, time.time() - tic)
            )
            autotvm.callback.log_to_file(log_filename)(None, inputs, results)


def evaluate(lib, ctx, name_n_data, dtype):
    # Setup runtime module.
    mod = runtime.GraphModule(lib["default"](ctx))
    for name, data in name_n_data.items():
        mod.set_input(name, data)

    # Evaluate performance.
    sys.stderr.write("Evaluate inference time cost...\n")
    ftimer = mod.module.time_evaluator("run", ctx, number=5, min_repeat_ms=1000)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    sys.stderr.write("Median inference time: %.2f ms\n" % np.median(prof_res))


def tune_and_evaluate(
    mod, params, inputs, dtype, measure_top_n, target, tuning_opt, ref_log_filename
):
    """Tune a model with the ranking model and evaluate the performance."""

    sys.stderr.write("Extract conv2d tasks...\n")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    tic = time.time()
    dispatch_ctx = tvm.autotvm.task.DispatchContext.current

    if not os.path.exists(tuning_opt["log_filename"]):
        tune_kernels(tasks, measure_top_n, **tuning_opt)
        sys.stderr.write("Tuning time: %.2f mins\n" % ((time.time() - tic) / 60))

    sys.stderr.write("Compile new tuned model...\n")
    tvm.autotvm.task.DispatchContext.current = autotvm.apply_history_best(
        tuning_opt["log_filename"]
    )

    compile_engine.get().clear()
    with relay.build_config(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)
    tvm.autotvm.task.DispatchContext.current = dispatch_ctx

    sys.stderr.write("Compile baseline model...\n")
    compile_engine.get().clear()
    with relay.build_config(opt_level=3):
        base_lib = relay.build_module.build(mod, target=target, params=params)

    ref_lib = None
    if os.path.exists(ref_log_filename):
        sys.stderr.write("Compile reference model...\n")
        tvm.autotvm.task.DispatchContext.current = autotvm.apply_history_best(ref_log_filename)

        compile_engine.get().clear()
        with relay.build_config(opt_level=3):
            ref_lib = relay.build_module.build(mod, target=target, params=params)
        tvm.autotvm.task.DispatchContext.current = dispatch_ctx

    ctx = tvm.context(str(target), 0)
    data_tvm = {}
    for name, shape in inputs.items():
        data_tvm[name] = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype))

    # Evaluate baseline module
    sys.stderr.write("Baseline\n")
    evaluate(base_lib, ctx, data_tvm, dtype)

    # Evaluate reference module
    if ref_lib:
        sys.stderr.write("Reference\n")
        evaluate(ref_lib, ctx, data_tvm, dtype)

    # Evaluate module
    evaluate(lib, ctx, data_tvm, dtype)


def main():
    """Main entry."""
    configs = create_config()
    np.random.seed(configs.seed)
    random.seed(configs.seed)

    # Check if the target is for x86.
    target = tvm.target.Target(configs.target)
    measure_cfgs = {
        "number": 5,
        "repeat": 1,
        "min_repeat_ms": 1000,
        "enable_cpu_cache_flush": False,
    }

    # Process shape
    if configs.mx is not None:
        assert configs.mx in model_shape_dict, "Model not found: {}".format(configs.mx)
        models = [configs.mx]
    else:
        models = list(model_shape_dict.keys())[5:6]

    for model in models:
        sys.stderr.write("%s\n" % model)

        # Get the model.
        mod, params, inputs = get_mxnet_network(model, model_shape_dict[model])
    
        # Map from task name to model.
        models = {}
        if configs.list_net is not None:
            for model_path in glob.glob("{}/*".format(configs.list_net)):
                if not os.path.isdir(model_path):
                    continue
                task_name = os.path.basename(model_path)
                models[task_name] = NNRankModel(task_name, model_path)
                print("Loaded cost model for %s" % task_name)
    
        measure_option = autotvm.measure_option(
            builder=DummyBuilder(configs.n_parallel),
            runner=RankModelRunner(models=models, **measure_cfgs),
        )
        tuning_option = {
            "log_filename": "{}_tune.log".format(model),
            "tuner": "round",
            "early_stopping": None,
            "n_trial": configs.n_trial,
            "measure_option": measure_option,
        }
    
        tune_and_evaluate(
            mod,
            params,
            inputs,
            "float32",
            configs.measure_top_n,
            configs.target,
            tuning_option,
            "{}_history_best.log".format(model),
        )


if __name__ == "__main__":
    main()
