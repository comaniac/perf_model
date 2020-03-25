import argparse
import pandas as pd
import numpy as np
import autogluon
from autogluon import TabularPrediction as task

INVALID_THD = 10  # Invalid throughput threshold ratio.
INVALID_LOG_THD = np.log(INVALID_THD)


def parse_args():
    parser = argparse.ArgumentParser(description='Cost Model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input csv file.')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of the test data.')
    args = parser.parse_args()
    return args


def train(args):
    df = pd.read_csv(args.dataset)
    # Pre-filter the invalid through-puts.
    # For these through-puts, we can directly obtain the result from the
    df = df[df['thrpt'] >= INVALID_THD]
    # Split Train/Test
    num_test = int(len(df) * args.test_ratio)
    num_train = len(df) - num_test
    train_df = df[:num_train]
    test_df = df[num_train:]

    predictor = task.fit(train_data=task.Dataset(df=train_df), label='thrpt')
    performance = predictor.evaluate(test_df)
    test_prediction = predictor.predict(test_df)
    print(performance)


if __name__ == "__main__":
    args = parse_args()
    train(args)
