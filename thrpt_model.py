import argparse
import pandas as pd
import numpy as np
import autogluon
import os
from autogluon import TabularPrediction as task

INVALID_THD = 10  # Invalid throughput threshold ratio.
INVALID_LOG_THD = np.log(INVALID_THD)


def parse_args():
    parser = argparse.ArgumentParser(description='Cost Model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input csv file.')
    parser.add_argument('--out_dir', type=str, default='thrpt_model_out',
                        help='output path of the througput model.')
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

    predictor = task.fit(train_data=task.Dataset(df=train_df),
                         output_directory=args.out_dir, label='thrpt')
    performance = predictor.evaluate(test_df)
    test_prediction = predictor.predict(test_df)
    ret = []
    for i, (lhs, rhs) in enumerate(zip(test_df['thrpt'].to_numpy(), test_prediction)):
        ret.append((i, lhs, rhs))
    df_result = pd.DataFrame(ret)
    df_result.to_csv(os.path.join(args.out_dir, 'pred_result.csv'))


if __name__ == "__main__":
    args = parse_args()
    train(args)
