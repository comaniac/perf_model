import argparse
import pandas as pd
import numpy as np
import autogluon
import os
from autogluon import TabularPrediction as task
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import matplotlib.pyplot as plt

INVALID_THD = 10  # Invalid throughput threshold ratio.
INVALID_LOG_THD = np.log(INVALID_THD)


def plot_save_figure(gt_thrpt, pred_thrpt, save_dir=None):
    plt.close()
    fig, ax = plt.subplots()
    xmin, xmax = gt_thrpt.min(), gt_thrpt.max()
    ymin, ymax = min(xmin, pred_thrpt.min()), max(xmax, pred_thrpt.max())
    ax.set(xlim=(xmin * 0.9, xmax * 1.1), ylim=(ymin * 0.9, ymax * 1.1))
    ax.set(xlabel='Ground-Truth Throughput', ylabel='Predicted Throughput',
           title='Throughput model error analysis')
    ax.plot(sorted(gt_thrpt), sorted(gt_thrpt), linewidth=1, label='Golden Line')
    ax.scatter(gt_thrpt, pred_thrpt, 0.5, color='red', label='Prediction')
    ax.legend()
    if save_dir is not None:
        out_path = os.path.join(save_dir, 'pred_result.png')
    else:
        out_path = 'pred_result.png'
    fig.savefig(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Cost Model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input csv file.')
    parser.add_argument('--out_dir', type=str, default='thrpt_model_out',
                        help='output path of the througput model.')
    parser.add_argument('--auto', action='store_true',
                        help='Whether to use AutoGluon.')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of the test data.')
    parser.add_argument('--lr', type=float, default=1E-3,
                        help='The learning rate of the throuphput model.')
    parser.add_argument('--wd', type=float, default=1E-5,
                        help='The weight decay of the throuphput model.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='The batch size')
    parser.add_argument('--niter', type=int, default=10000,
                        help='The total number of training iterations.')
    parser.add_argument('--nval_iter', type=int, default=1000,
                        help='The number of validation iterations.')
    parser.add_argument('--delta_thrpt', type=float, default=5)
    args = parser.parse_args()
    return args


def get_data(args):
    df = pd.read_csv(args.dataset)
    # Pre-filter the invalid through-puts.
    # For these through-puts, we can directly obtain the result from the
    df = df[df['thrpt'] >= INVALID_THD]
    # Split Train/Test
    num_train = int(len(df) * (1 - args.test_ratio))
    train_df = df[:num_train]
    test_df = df[num_train:]
    return train_df, test_df


def train_regression_autogluon(train_df, test_df):
    predictor = task.fit(train_data=task.Dataset(df=train_df),
                         output_directory=args.out_dir, label='thrpt')
    performance = predictor.evaluate(test_df)
    test_prediction = predictor.predict(test_df)
    ret = np.zeros((len(test_prediction), 2), dtype=np.float32)
    for i, (lhs, rhs) in enumerate(zip(test_df['thrpt'].to_numpy(), test_prediction)):
        ret[i][0] = lhs
        ret[i][1] = rhs
    df_result = pd.DataFrame(ret, columns=['gt', 'pred'])
    df_result.to_csv(os.path.join(args.out_dir, 'pred_result.csv'))
    plot_save_figure(gt_thrpt=test_df['thrpt'].to_numpy(),
                     pred_thrpt=test_prediction)


class PerfNet(gluon.HybridBlock):
    def __init__(self, num_hidden=64, num_layer=2, dropout=0.1, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.layers = nn.HybridSequential()
            with self.layers.name_scope():
                for i in range(num_layer):
                    self.layers.add(nn.Dense(64, flatten=False))
                    if i != num_layer - 1:
                        self.layers.add(nn.LeakyReLU(0.1))
                        self.layers.add(nn.Dropout(dropout))

    def hybrid_forward(self, F, data):
        return self.layers(data)


def evaluate_ranking(test_df):
    pass


def train_ranking(train_df, test_df):
    embed_net = PerfNet()
    rank_score_net = nn.Dense(1, flatten=False)
    regression_score_net = nn.Dense(1, flatten=False)
    feature_keys = [ele for ele in train_df.keys() if ele != 'thrpt']
    train_features = train_df[feature_keys].to_numpy()
    train_labels = train_df['thrpt'].to_numpy()
    test_features = test_df[feature_keys].to_numpy()
    test_labels = test_df['thrpt'].to_numpy()
    train_feature_mean = train_features.mean(axis=1)
    train_feature_std = train_features.std(axis=1)
    loss = gluon.loss.HuberLoss()
    optimizer_params = {'learning_rate': args.lr,
                        'wd': args.wd}
    embed_trainer = gluon.Trainer(embed_net.collect_params(), 'adam', optimizer_params)
    rank_score_trainer = gluon.Trainer(rank_score_net.collect_params(), 'adam',
                                       optimizer_params)
    rank_score_trainer = gluon.Trainer(rank_score_net.collect_params(), 'adam',
                                       optimizer_params)
    for i in range(args.niter):
        with mx.autograd.record():
            pass


if __name__ == "__main__":
    args = parse_args()
    train_df, test_df = get_data(args)
    train_regression_autogluon(train_df, test_df)
