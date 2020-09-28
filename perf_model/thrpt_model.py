# pylint: disable=missing-docstring, invalid-name, ungrouped-imports
# pylint: disable=unsupported-assignment-operation, no-member
import argparse
import logging
import os
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
import tqdm
from mxnet import gluon
from mxnet.gluon import nn
from sklearn.metrics import ndcg_score

from util import analyze_valid_threshold, grad_global_norm, logging_config, parse_ctx, set_seed

mx.npx.set_np()


def plot_save_figure(gt_thrpt, pred_thrpt, save_dir=None):
    plt.close()
    fig, ax = plt.subplots()
    xmin, xmax = gt_thrpt.min(), gt_thrpt.max()
    ymin, ymax = min(xmin, pred_thrpt.min()), max(xmax, pred_thrpt.max())
    ax.set(xlim=(xmin * 0.9, xmax * 1.1), ylim=(ymin * 0.9, ymax * 1.1))
    ax.set(xlabel='Ground-Truth Throughput',
           ylabel='Predicted Throughput',
           title='Throughput model error analysis')
    ax.plot(sorted(gt_thrpt),
            sorted(gt_thrpt),
            linewidth=1,
            label='Golden Line')
    ax.scatter(gt_thrpt, pred_thrpt, 0.5, color='red', label='Prediction')
    ax.legend()
    if save_dir is not None:
        out_path = os.path.join(save_dir, 'pred_result.png')
    else:
        out_path = 'pred_result.png'
    fig.savefig(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Cost Model')
    parser.add_argument('--seed',
                        type=int,
                        default=100,
                        help='Seed for the training.')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='path to the input csv file.')
    parser.add_argument('--out_dir',
                        type=str,
                        default='thrpt_model_out',
                        help='output path of the througput model.')
    parser.add_argument('--algo',
                        choices=['auto', 'cat', 'nn'],
                        default='nn',
                        help='The algorithm to use.')
    parser.add_argument('--test_ratio',
                        type=float,
                        default=0.1,
                        help='ratio of the test data.')
    parser.add_argument('--dev_ratio',
                        type=float,
                        default=0.1,
                        help='ratio of the test data.')
    rank_args = parser.add_argument_group('ranking')
    rank_args.add_argument('--num_threshold_bins', default=5, type=int)
    rank_args.add_argument('--group_size', default=40, type=int)
    rank_args.add_argument('--sample_mult', default=5, type=int)
    rank_args.add_argument('--rank_loss_function',
                           choices=['YetiRank', 'YetiRankPairwise'])
    rank_args.add_argument('--test_seed', default=123, type=int)
    rank_args.add_argument('--dev_seed', default=1234, type=int)
    parser.add_argument('--lr',
                        type=float,
                        default=1E-3,
                        help='The learning rate of the throuphput model.')
    parser.add_argument('--wd',
                        type=float,
                        default=0.0,
                        help='The weight decay of the throuphput model.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='The batch size')
    parser.add_argument('--rank_npair',
                        type=int,
                        default=20,
                        help='How many pairs to compare in the ranking loss.')
    parser.add_argument('--niter',
                        type=int,
                        default=10000000,
                        help='The total number of training iterations.')
    parser.add_argument('--log_interval',
                        type=int,
                        default=1000,
                        help='The logging interval.')
    parser.add_argument('--nval_iter',
                        type=int,
                        default=10000,
                        help='The number of iterations per validation.')
    parser.add_argument('--regress_alpha',
                        type=float,
                        default=1.0,
                        help='Control the weight of the regression loss')
    parser.add_argument('--rank_alpha',
                        type=float,
                        default=1.0,
                        help='Control the weight of the ranking loss')
    parser.add_argument('--threshold', type=float, default=5)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    parser.add_argument(
        '--thrpt-threshold',
        type=float,
        default=0,
        help='The throughput threshold. -1 means adaptive threshold')
    parser.add_argument(
        '--max-data-size',
        type=int,
        default=524288,
        help='The maximum number of records in dataset')
    args = parser.parse_args()
    return args


def get_data(args):
    if args.thrpt_threshold == -1:
        invalid_thd = analyze_valid_threshold(args.dataset)
    else:
        invalid_thd = 0

    df = pd.read_csv(args.dataset)
    # Pre-filter the invalid through-puts.
    # For these throughputs, we can directly obtain the result from the ValidNet
    logging.info('Invalid throughput is set to %.1f GFLOP/s', invalid_thd)
    df = df[df['thrpt'] >= invalid_thd]
    assert df.shape[0] > 0

    used_keys = []
    not_used_keys = []
    for key in df.keys():
        if key == 'thrpt':
            used_keys.append(key)
            continue
        if df[key].to_numpy().std() == 0:
            not_used_keys.append(key)
            continue
        used_keys.append(key)
    logging.info('Original keys=%s, Not used keys=%s', list(df.keys()),
                 not_used_keys)
    df = df[used_keys]

    # Sampling if data set is too large
    # if args.max_data_size < len(df):
    #     df = df.sample(n=args.max_data_size, random_state=11)

    # Split Train/Test
    num_train = int(len(df) * (1 - args.test_ratio))
    train_df = df[:num_train]
    test_df = df[num_train:]
    return train_df, test_df


def get_feature_label(df):
    feature_keys = [ele for ele in df.keys() if ele != 'thrpt']
    features = df[feature_keys].to_numpy()
    labels = df['thrpt'].to_numpy()
    return features, labels


def train_regression_autogluon(args, train_df, test_df):
    mx.npx.reset_np()
    from autogluon import TabularPrediction as task
    predictor = task.fit(train_data=task.Dataset(df=train_df),
                         output_directory=args.out_dir,
                         label='thrpt',
                         eval_metric='mean_absolute_error')
    #performance = predictor.evaluate(test_df)
    test_prediction = predictor.predict(test_df)
    ret = np.zeros((len(test_prediction), 2), dtype=np.float32)
    for i, (lhs,
            rhs) in enumerate(zip(test_df['thrpt'].to_numpy(),
                                  test_prediction)):
        ret[i][0] = lhs
        ret[i][1] = rhs
    df_result = pd.DataFrame(ret, columns=['gt', 'pred'])
    df_result.to_csv(os.path.join(args.out_dir, 'pred_result.csv'))
    plot_save_figure(gt_thrpt=test_df['thrpt'].to_numpy(),
                     pred_thrpt=test_prediction,
                     save_dir=args.out_dir)
    mx.npx.set_np()


class EmbedNet(gluon.HybridBlock):
    def __init__(self,
                 num_hidden=64,
                 num_layer=2,
                 dropout=0.1,
                 prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.layers = nn.HybridSequential()
            with self.layers.name_scope():
                for _ in range(num_layer):
                    self.layers.add(nn.Dense(num_hidden, flatten=False))
                    self.layers.add(nn.LeakyReLU(0.1))
                    self.layers.add(nn.Dropout(dropout))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.layers(x)


def get_nn_pred_scores(embed_net, regression_thrpt_net, test_features,
                       feat_mean, feat_std, batch_size):
    pred_scores = []
    for i in range(0, test_features.shape[0], batch_size):
        batch_feature = (test_features[i:(i + batch_size)] -
                         feat_mean) / feat_std
        embeddings = embed_net(batch_feature)
        scores = regression_thrpt_net(embeddings)
        pred_scores.extend(scores.asnumpy().to_list())
    return pred_scores


def evaluate_nn(test_features, test_labels, embed_net, regression_thrpt_net,
                rank_score_net, batch_size, num_hidden, ctx, threshold):
    n_samples = test_features.shape[0]
    embeddings = mx.np.zeros(shape=(n_samples, num_hidden),
                             dtype=np.float32,
                             ctx=ctx)
    for i in range(0, n_samples, batch_size):
        batch_features = mx.np.array(test_features[i:(i + batch_size)],
                                     dtype=np.float32,
                                     ctx=ctx)
        embeddings[i:(i + batch_size), :] = embed_net(batch_features)
    # Calculate regression scores
    total_mae = 0
    total_mae_cnt = 0
    for i in range(0, n_samples, batch_size):
        batch_embeddings = embeddings[i:(i + batch_size)]
        batch_labels = test_labels[i:(i + batch_size)]
        batch_pred_thrpt = regression_thrpt_net(batch_embeddings)[:, 0]
        total_mae += mx.np.abs(
            batch_pred_thrpt -
            mx.np.array(batch_labels, dtype=np.float32, ctx=ctx)).sum()
        total_mae_cnt += batch_labels.shape[0]
    # Calculate ranking scores
    n_correct = 0
    n_total = 0
    total_nll = 0
    gt_label_distribution = np.zeros(shape=(3, ), dtype=np.int64)
    for i in range(0, n_samples, batch_size):
        lhs_embeddings = embeddings[i:(i + batch_size)]
        lhs_labels = test_labels[i:(i + batch_size)]
        for j in range(0, n_samples, batch_size):
            rhs_embeddings = embeddings[j:(j + batch_size)]
            rhs_labels = test_labels[j:(j + batch_size)]
            pair_scores = np.expand_dims(lhs_labels, axis=1) - np.expand_dims(
                rhs_labels, axis=0)
            pair_label = np.zeros_like(pair_scores, dtype=np.int32)
            pair_label[pair_scores >= threshold] = 1
            pair_label[pair_scores <= -threshold] = 2
            pair_label = mx.np.array(pair_label, dtype=np.int32, ctx=ctx)
            inner_lhs_embeddings = mx.np.expand_dims(lhs_embeddings, axis=1)
            rhs_embeddings = mx.np.expand_dims(rhs_embeddings, axis=0)
            inner_lhs_embeddings, rhs_embeddings =\
                mx.np.broadcast_arrays(inner_lhs_embeddings, rhs_embeddings)
            joint_embedding = mx.np.concatenate([
                inner_lhs_embeddings, rhs_embeddings,
                mx.np.abs(inner_lhs_embeddings - rhs_embeddings),
                inner_lhs_embeddings * rhs_embeddings
            ],
                                                axis=-1)
            pred_rank_label_scores = rank_score_net(joint_embedding)
            logits = mx.npx.pick(
                mx.npx.log_softmax(pred_rank_label_scores, axis=-1),
                pair_label)
            n_correct += (pred_rank_label_scores.argmax(axis=-1).astype(
                np.int32) == pair_label).sum()
            n_total += np.prod(pair_label.shape)
            total_nll += -logits.sum()
            for k in range(3):
                gt_label_distribution[k] += (pair_label == k).sum()
    return (total_nll / n_total, total_mae / total_mae_cnt,
            n_correct / n_total, n_correct, n_total, gt_label_distribution)


def train_nn(args, train_df, test_df):
    ctx = parse_ctx(args.gpus)[0]
    batch_size = args.batch_size
    embed_net = EmbedNet(args.num_hidden,
                         args.num_layers,
                         args.dropout,
                         prefix='embed_net_')
    rank_score_net = nn.HybridSequential(prefix='rank_score_net_')
    with rank_score_net.name_scope():
        rank_score_net.add(nn.Dense(args.num_hidden, flatten=False))
        rank_score_net.add(nn.LeakyReLU(0.1))
        rank_score_net.add(nn.Dense(3, flatten=False))
    regression_thrpt_net = nn.HybridSequential(prefix='regression_thrpt_net_')
    with regression_thrpt_net.name_scope():
        regression_thrpt_net.add(nn.Dense(args.num_hidden, flatten=False))
        regression_thrpt_net.add(nn.LeakyReLU(0.1))
        regression_thrpt_net = nn.Dense(1, flatten=False)
    embed_net.hybridize()
    rank_score_net.hybridize()
    regression_thrpt_net.hybridize()
    rank_loss_func = gluon.loss.SoftmaxCrossEntropyLoss(batch_axis=[0, 1])

    train_dev_features, train_dev_labels = get_feature_label(train_df)
    shuffle_idx = np.random.permutation(train_dev_features.shape[0])
    train_dev_features, train_dev_labels =\
        train_dev_features[shuffle_idx], train_dev_labels[shuffle_idx]
    num_train = train_dev_features.shape[0] - int(
        args.dev_ratio * train_dev_features.shape[0])
    train_features, train_labels = train_dev_features[:num_train], train_dev_labels[:num_train]
    dev_features, dev_labels = train_dev_features[
        num_train:], train_dev_labels[num_train:]
    test_features, test_labels = get_feature_label(test_df)
    embed_net.initialize(init=mx.init.Normal(0.01), ctx=ctx)
    rank_score_net.initialize(init=mx.init.Normal(0.01), ctx=ctx)
    regression_thrpt_net.initialize(init=mx.init.Normal(0.01), ctx=ctx)
    optimizer_params = {'learning_rate': args.lr, 'wd': args.wd}
    embed_trainer = gluon.Trainer(embed_net.collect_params(), 'adam',
                                  optimizer_params)
    rank_score_trainer = gluon.Trainer(rank_score_net.collect_params(), 'adam',
                                       optimizer_params)
    regression_score_trainer = gluon.Trainer(
        regression_thrpt_net.collect_params(), 'adam', optimizer_params)
    avg_regress_loss = 0
    avg_regress_loss_denom = 0
    avg_rank_loss = 0
    avg_rank_loss_denom = 0
    avg_embed_net_norm = 0
    avg_rank_score_net_norm = 0
    avg_regression_thrpt_net_norm = 0
    avg_norm_iter = 0
    best_val_loss = np.inf
    no_better_val_cnt = 0
    curr_lr = args.lr
    best_val_loss_f = open(os.path.join(args.out_dir, 'best_val_acc.csv'), 'w')
    test_loss_f = open(os.path.join(args.out_dir, 'test_acc.csv'), 'w')
    for i in range(args.niter):
        # Sample random minibatch
        # We can later revise the algorithm to use stratified sampling
        sample_idx = np.random.randint(0, train_features.shape[0], batch_size)
        pair_sample_idx = np.random.randint(0, train_features.shape[0],
                                            (batch_size, args.rank_npair))
        # Shape (batch_size, C), (batch_size)
        batch_feature, batch_label = np.take(train_features, sample_idx, axis=0),\
                                     np.take(train_labels, sample_idx, axis=0)
        rank_pair_feature, rank_pair_label = np.take(train_features, pair_sample_idx, axis=0),\
                                             np.take(train_labels, pair_sample_idx, axis=0)
        # 1. -threshold < score_1 - score2 < threshold, rank_label = 0
        # 2. score_1 - score2 >= threshold, rank_label = 1
        # 3. score_1 - score2 <= -threshold, rank_label = 2
        # (i, j) --> score_i - score_j
        pair_score = np.expand_dims(batch_label, axis=1) - rank_pair_label
        pair_label = np.zeros_like(pair_score, dtype=np.int32)
        pair_label[pair_score >= args.threshold] = 1
        pair_label[pair_score <= -args.threshold] = 2
        batch_feature = mx.np.array(batch_feature, dtype=np.float32, ctx=ctx)
        batch_label = mx.np.array(batch_label, dtype=np.float32, ctx=ctx)
        rank_pair_feature = mx.np.array(rank_pair_feature,
                                        dtype=np.float32,
                                        ctx=ctx)
        pair_label = mx.np.array(pair_label, dtype=np.int32, ctx=ctx)
        with mx.autograd.record():
            lhs_embedding = embed_net(batch_feature)
            pred_score = regression_thrpt_net(lhs_embedding)[:, 0]
            regress_loss = mx.np.abs(pred_score - batch_label).mean()

            rhs_embedding = embed_net(rank_pair_feature)
            # Concatenate the embedding
            lhs_embedding = mx.np.expand_dims(lhs_embedding, axis=1)
            lhs_embedding = mx.np.broadcast_to(lhs_embedding,
                                               rhs_embedding.shape)
            joint_embedding = mx.np.concatenate([
                lhs_embedding, rhs_embedding,
                mx.np.abs(lhs_embedding - rhs_embedding),
                lhs_embedding * rhs_embedding
            ],
                                                axis=-1)
            rank_logits = rank_score_net(joint_embedding)
            rank_loss = rank_loss_func(rank_logits, pair_label).mean()
            loss = args.regress_alpha * regress_loss + args.rank_alpha * rank_loss
        loss.backward()
        embed_net_norm = grad_global_norm(embed_net.collect_params().values())
        rank_score_net_norm = grad_global_norm(
            rank_score_net.collect_params().values())
        regression_thrpt_net_norm = grad_global_norm(
            regression_thrpt_net.collect_params().values())
        avg_embed_net_norm += embed_net_norm
        avg_rank_score_net_norm += rank_score_net_norm
        avg_regression_thrpt_net_norm += regression_thrpt_net_norm
        avg_norm_iter += 1
        embed_trainer.step(1.0)
        rank_score_trainer.step(1.0)
        regression_score_trainer.step(1.0)
        avg_regress_loss += regress_loss.asnumpy() * batch_size
        avg_regress_loss_denom += batch_size
        avg_rank_loss += rank_loss.asnumpy() * batch_size * batch_size
        avg_rank_loss_denom += batch_size * batch_size
        if (i + 1) % args.log_interval == 0:
            logging.info(
                'Iter:%d/%d, Train Loss Regression/Ranking=%f/%f, '
                'grad_norm Embed/Regression/Rank=%f/%f/%f', i + 1, args.niter,
                avg_regress_loss / avg_regress_loss_denom,
                avg_rank_loss / avg_rank_loss_denom,
                avg_embed_net_norm / avg_norm_iter,
                avg_regression_thrpt_net_norm / avg_norm_iter,
                avg_rank_score_net_norm / avg_norm_iter)
            avg_regress_loss = 0
            avg_regress_loss_denom = 0
            avg_rank_loss = 0
            avg_rank_loss_denom = 0
            avg_embed_net_norm = 0
            avg_rank_score_net_norm = 0
            avg_regression_thrpt_net_norm = 0
            avg_norm_iter = 0
        if (i + 1) % args.nval_iter == 0:
            val_nll, val_mae, val_acc, n_correct, n_total, gt_label_distribution =\
                evaluate_nn(dev_features, dev_labels, embed_net, regression_thrpt_net,
                            rank_score_net, batch_size, args.num_hidden, ctx, args.threshold)
            pair_label_total = gt_label_distribution.sum()
            logging.info(
                'Validation error: nll=%f, mae=%f, acc=%f,'
                ' correct/total=%d/%d,'
                ' dev distribution equal: %.2f, lhs>rhs: %.2f, lhs<rhs: %.2f',
                val_nll, val_mae, val_acc, n_correct, n_total,
                gt_label_distribution[0] / pair_label_total * 100,
                gt_label_distribution[1] / pair_label_total * 100,
                gt_label_distribution[2] / pair_label_total * 100)
            val_loss = args.regress_alpha * val_nll + args.rank_alpha * val_mae
            if val_loss < best_val_loss:
                no_better_val_cnt = 0
                best_val_loss = val_loss
                embed_net.export(os.path.join(args.out_dir, 'embed_net'))
                rank_score_net.export(
                    os.path.join(args.out_dir, 'rank_score_net'))
                best_val_loss_f.write('{}, {}, {}, {}\n'.format(
                    i + 1, val_acc, val_mae, val_nll))
                best_val_loss_f.flush()
                (test_nll, test_mae, test_acc, test_n_correct, test_n_total,
                 test_gt_label_distribution) = evaluate_nn(
                     test_features, test_labels, embed_net,
                     regression_thrpt_net, rank_score_net, batch_size,
                     args.num_hidden, ctx, args.threshold)
                test_loss_f.write('{}, {}, {}, {}\n'.format(
                    i + 1, test_acc, test_mae, test_nll))
                test_loss_f.flush()
                logging.info(
                    'Test error: nll=%f, mae=%f, acc=%f,'
                    ' correct/total=%d/%d,'
                    ' test distribution equal: %.2f, lhs>rhs: %.2f, lhs<rhs: %.2f',
                    test_nll, test_mae, test_acc, test_n_correct, test_n_total,
                    test_gt_label_distribution[0] /
                    test_gt_label_distribution.sum() * 100,
                    test_gt_label_distribution[1] /
                    test_gt_label_distribution.sum() * 100,
                    test_gt_label_distribution[2] /
                    test_gt_label_distribution.sum() * 100)
            else:
                no_better_val_cnt += 1
                if no_better_val_cnt > 5:
                    if curr_lr == 1E-5:
                        break
                    curr_lr = max(curr_lr / 2, 1E-5)
                    embed_trainer.set_learning_rate(curr_lr)
                    regression_score_trainer.set_learning_rate(curr_lr)
                    rank_score_trainer.set_learning_rate(curr_lr)
                    logging.info('Decrease learning rate to %f', curr_lr)
                    no_better_val_cnt = 0
    best_val_loss_f.close()
    test_loss_f.close()


def train_ranking_catboost(args, train_df, test_df):
    import catboost
    params = {
        'loss_function': args.rank_loss_function,
        'custom_metric': ['NDCG', 'AverageGain:top=10'],
        'task_type': 'GPU',
        'iterations': args.niter,
        'verbose': True,
        'train_dir': args.out_dir,
        'random_seed': args.seed
    }
    train_dev_features, train_dev_labels = get_feature_label(train_df)
    test_features, test_labels = get_feature_label(test_df)

    # Split Train/Dev
    shuffle_idx = np.random.permutation(train_dev_features.shape[0])
    train_dev_features, train_dev_labels = \
        train_dev_features[shuffle_idx], train_dev_labels[shuffle_idx]
    num_train = train_dev_features.shape[0] - int(
        args.dev_ratio * train_dev_features.shape[0])
    train_features, train_labels = train_dev_features[:num_train], train_dev_labels[:num_train]
    dev_features, dev_labels = train_dev_features[num_train:], train_dev_labels[num_train:]

    total_data_size = len(train_df) + len(test_df)
    get_sample_size = lambda ratio: \
        int(min(total_data_size, args.max_data_size) * ratio * args.sample_mult)
    dev_sample_size = get_sample_size(args.dev_ratio * (1 - args.test_ratio))
    train_sample_size = get_sample_size((1 - args.dev_ratio) * (1 - args.test_ratio))
    test_sample_size = get_sample_size(args.test_ratio)

    # Generate the training/testing samples for ranking.
    # We divide the samples into multiple bins and will do stratified sampling within each bin.
    sorted_train_ids = np.argsort(train_labels)
    train_group_ids_list = np.array_split(sorted_train_ids, args.num_threshold_bins)

    sorted_dev_ids = np.argsort(dev_labels)
    dev_group_ids_list = np.array_split(sorted_dev_ids, args.num_threshold_bins)

    sorted_test_ids = np.argsort(test_labels)
    test_group_ids_list = np.array_split(sorted_test_ids, args.num_threshold_bins)

    train_rank_features = []
    train_rank_labels = []
    train_groups = []

    dev_rank_features = []
    dev_rank_labels = []
    dev_groups = []

    test_rank_features = []
    test_rank_labels = []
    test_groups = []

    train_npz_file = os.path.join(args.out_dir, 'train_rank_features.npz')
    dev_npz_file = os.path.join(args.out_dir, 'dev_rank_features.npz')
    test_npz_file = os.path.join(args.out_dir, 'test_rank_features.npz')
    if os.path.exists(test_npz_file):
        print('Loading features from npz')
        assert os.path.exists(train_npz_file)
        assert os.path.exists(dev_npz_file)

        npzfile = np.load(train_npz_file)
        train_rank_features = npzfile['train_rank_features']
        train_rank_labels = npzfile['train_rank_labels']
        train_groups = npzfile['train_groups']

        npzfile = np.load(dev_npz_file)
        dev_rank_features = npzfile['dev_rank_features']
        dev_rank_labels = npzfile['dev_rank_labels']
        dev_groups = npzfile['dev_groups']

        npzfile = np.load(test_npz_file)
        test_rank_features = npzfile['test_rank_features']
        test_rank_labels = npzfile['test_rank_labels']
        test_groups = npzfile['test_groups']
    else:

        print('Generate Dev Ranking Groups:')
        for i in tqdm.tqdm(range(dev_sample_size)):
            for group_ids in dev_group_ids_list:
                chosen_ids = np.random.choice(group_ids,
                                              args.group_size // args.num_threshold_bins,
                                              replace=False)
                dev_rank_features.append(dev_features[chosen_ids, :])
                dev_rank_labels.append(dev_labels[chosen_ids])
                dev_groups.append(np.ones_like(chosen_ids) * i)
        dev_rank_features = np.concatenate(dev_rank_features, axis=0)
        dev_rank_labels = np.concatenate(dev_rank_labels, axis=0)
        dev_groups = np.concatenate(dev_groups, axis=0)

        np.savez(os.path.join(args.out_dir, 'dev_rank_features.npz'),
                 dev_rank_features=dev_rank_features,
                 dev_rank_labels=dev_rank_labels,
                 dev_groups=dev_groups)

        print('Generate Train Ranking Groups:')
        for i in tqdm.tqdm(range(train_sample_size)):
            for group_ids in train_group_ids_list:
                chosen_ids = np.random.choice(group_ids,
                                              args.group_size // args.num_threshold_bins,
                                              replace=False)
                train_rank_features.append(train_features[chosen_ids, :])
                train_rank_labels.append(train_labels[chosen_ids])
                train_groups.append(np.ones_like(chosen_ids) * i)
        train_rank_features = np.concatenate(train_rank_features, axis=0)
        train_rank_labels = np.concatenate(train_rank_labels, axis=0)
        train_groups = np.concatenate(train_groups, axis=0)

        np.savez(os.path.join(args.out_dir, 'train_rank_features.npz'),
                 train_rank_features=train_rank_features,
                 train_rank_labels=train_rank_labels,
                 train_groups=train_groups)

        test_rng = np.random.RandomState(args.test_seed)
        print('Generate Test Ranking Groups:')
        for i in tqdm.tqdm(range(test_sample_size)):
            for group_ids in test_group_ids_list:
                chosen_ids = test_rng.choice(group_ids,
                                             args.group_size // args.num_threshold_bins,
                                             replace=False)
                test_rank_features.append(test_features[chosen_ids, :])
                test_rank_labels.append(test_labels[chosen_ids])
                test_groups.append(np.ones_like(chosen_ids) * i)
        test_rank_features = np.concatenate(test_rank_features, axis=0)
        test_rank_labels = np.concatenate(test_rank_labels, axis=0)
        test_groups = np.concatenate(test_groups, axis=0)

        np.savez(os.path.join(args.out_dir, 'test_rank_features.npz'),
                 test_rank_features=test_rank_features,
                 test_rank_labels=test_rank_labels,
                 test_groups=test_groups)

    train_pool = catboost.Pool(data=train_rank_features,
                               label=train_rank_labels,
                               group_id=train_groups)
    dev_pool = catboost.Pool(data=dev_rank_features,
                             label=dev_rank_labels,
                             group_id=dev_groups)
    # test_pool = catboost.Pool(data=test_rank_features,
    #                           label=test_rank_labels,
    #                           group_id=test_groups)
    model = catboost.CatBoost(params)
    model.fit(train_pool, eval_set=dev_pool)
    predict_result = model.predict(test_rank_features)

    test_gt_scores = test_rank_labels.reshape(test_sample_size, args.group_size)
    predict_result = predict_result.reshape((test_sample_size, args.group_size))
    np.save(os.path.join(args.out_dir, 'test_predictions.npy'), predict_result)
    test_ndcg_score = ndcg_score(y_true=test_gt_scores, y_score=predict_result)
    logging.info('Test NDCG=%f', test_ndcg_score)
    model.save_model(os.path.join(args.out_dir, 'list_rank_net.cbm'))
    model.save_model(os.path.join(args.out_dir, 'list_rank_net'), format='python')


def main():
    args = parse_args()
    set_seed(args.seed)
    logging_config(args.out_dir, 'thrpt_model')
    train_df, test_df = get_data(args)
    if args.algo == 'auto':
        train_regression_autogluon(args, train_df, test_df)
    elif args.algo == 'cat':
        train_ranking_catboost(args, train_df, test_df)
    elif args.algo == 'nn':
        train_nn(args, train_df, test_df)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
