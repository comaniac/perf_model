"""Util Functions."""
from math import ceil

import numpy as np
import pandas as pd

def analyze_valid_threshold(dataset):
    """Analyze throughputs and determine the invalid threshold."""
    MAX_INVALID_THD = 10

    df = pd.read_csv(dataset)
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
