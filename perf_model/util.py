"""Util Functions."""
from math import ceil
from collections import defaultdict
from typing import Optional
import inspect
import os
import numpy as np
import logging
import random
import pandas as pd


def read_pd(path):
    df = None
    for reader in [pd.read_csv, pd.read_parquet, pd.read_pickle]:
        try:
            df = reader(path)
            break
        except Exception:
            continue
    if df is None:
        raise RuntimeError('Cannot load {}'.format(path))
    return df


def analyze_valid_threshold(dataset):
    """Analyze throughputs and determine the invalid threshold."""
    MAX_INVALID_THD = 10

    df = read_pd(dataset)
    stats = pd.cut(x=df['thrpt'],
                   include_lowest=True,
                   bins=np.arange(0, ceil(max(df['thrpt'])) + 1, 0.1)).value_counts().sort_index()
    half_size = df.shape[0] // 2
    acc = 0
    for inv, val in zip(stats.index, stats):
        acc += val
        thd = inv.right.tolist()
        if acc >= half_size or thd >= MAX_INVALID_THD:
            return thd
    return -1


def logging_config(folder: Optional[str] = None,
                   name: Optional[str] = None,
                   level: int = logging.INFO,
                   console_level: int = logging.INFO,
                   console: bool = True) -> str:
    """Config the logging module"""
    if name is None:
        name = inspect.stack()[-1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to {}".format(logpath))
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def parse_ctx(data_str):
    import mxnet as mx
    if data_str == '-1' or data_str == '':
        ctx_l = [mx.cpu()]
    else:
        ctx_l = [mx.gpu(int(x)) for x in data_str.split(',')]
    return ctx_l


def set_seed(seed):
    import mxnet as mx
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)



def grad_global_norm(parameters) -> float:
    """Calculate the 2-norm of gradients of parameters, and how much they should be scaled down
    such that their 2-norm does not exceed `max_norm`, if `max_norm` if provided.
    If gradients exist for more than one context for a parameter, user needs to explicitly call
    ``trainer.allreduce_grads`` so that the gradients are summed first before calculating
    the 2-norm.

    .. note::
        This function is only for use when `update_on_kvstore` is set to False in trainer.

    Example::
        trainer = Trainer(net.collect_params(), update_on_kvstore=False, ...)
        for x, y in mx.gluon.utils.split_and_load(X, [mx.gpu(0), mx.gpu(1)]):
            with mx.autograd.record():
                y = net(x)
                loss = loss_fn(y, label)
            loss.backward()
        trainer.allreduce_grads()
        norm = grad_global_norm(net.collect_params().values())
        ...

    Parameters
    ----------
    parameters
        The list of Parameters

    Returns
    -------
    total_norm
        Total norm. It's a numpy scalar.
    """
    # Distribute gradients among contexts,
    # For example, assume there are 8 weights and four GPUs, we can ask each GPU to
    # compute the squared sum of two weights and then add the results together
    import mxnet as mx
    idx = 0
    arrays = defaultdict(list)
    sum_norms = []
    num_ctx = None
    for p in parameters:
        if p.grad_req != 'null':
            p_grads = p.list_grad()
            if num_ctx is None:
                num_ctx = len(p_grads)
            else:
                assert num_ctx == len(p_grads)
            arrays[idx % num_ctx].append(p_grads[idx % num_ctx])
            idx += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm.'

    # TODO(sxjscience)
    #  Investigate the float16 case.
    #  The inner computation accumulative type of norm should be float32.
    ctx = arrays[0][0].context
    for idx, arr_l in enumerate(arrays.values()):
        sum_norm = mx.np.linalg.norm(mx.np.concatenate([mx.np.ravel(ele) for ele in arr_l]))
        sum_norms.append(sum_norm.as_in_ctx(ctx))

    # Reduce over ctx
    if num_ctx == 1:
        total_norm = sum_norms[0]
    else:
        total_norm = mx.np.linalg.norm(mx.np.concatenate(sum_norms, axis=None))
    total_norm = float(total_norm)
    return total_norm

