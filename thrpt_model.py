import argparse
import pandas as pd
import numpy as np
import random
import logging
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from numpy_nlp.utils.parameter import grad_global_norm
from numpy_nlp.utils.misc import logging_config, set_seed, parse_ctx
import matplotlib.pyplot as plt

mx.npx.set_np()
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
    parser.add_argument('--seed', type=int, default=100, help='Seed for the training.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input csv file.')
    parser.add_argument('--out_dir', type=str, default='thrpt_model_out',
                        help='output path of the througput model.')
    parser.add_argument('--algo', choices=['auto', 'cat', 'nn'],
                        default='nn',
                        help='The algorithm to use.')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='ratio of the test data.')
    parser.add_argument('--dev_ratio', type=float, default=0.2,
                        help='ratio of the test data.')
    parser.add_argument('--lr', type=float, default=1E-3,
                        help='The learning rate of the throuphput model.')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='The weight decay of the throuphput model.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The batch size')
    parser.add_argument('--rank_npair', type=int, default=20,
                        help='How many pairs to compare in the ranking loss.')
    parser.add_argument('--niter', type=int, default=10000000,
                        help='The total number of training iterations.')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The logging interval.')
    parser.add_argument('--nval_iter', type=int, default=10000,
                        help='The number of iterations per validation.')
    parser.add_argument('--regress_alpha', type=float, default=0.0,
                        help='Control the weight of the regression loss')
    parser.add_argument('--rank_alpha', type=float, default=1.0,
                        help='Control the weight of the ranking loss')
    parser.add_argument('--threshold', type=float, default=5)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    args = parser.parse_args()
    return args


def get_data(args):
    df = pd.read_csv(args.dataset)
    # Pre-filter the invalid through-puts.
    # For these through-puts, we can directly obtain the result from the
    df = df[df['thrpt'] >= INVALID_THD]
    used_keys = []
    not_used_keys = []
    for key in df.keys():
        if key == 'thrpt':
            used_keys.append(key)
            continue
        if df[key].to_numpy().std() == 0:
            not_used_keys.append(key)
            continue
        else:
            used_keys.append(key)
    logging.info('Original keys={}, Not used keys={}'.format(list(df.keys()), not_used_keys))
    df = df[used_keys]
    # Split Train/Test
    num_train = int(len(df) * (1 - args.test_ratio))
    train_df = df[:num_train]
    test_df = df[num_train:]
    return train_df, test_df


def get_feature_label(df):
    feature_keys = [ele for ele in train_df.keys() if ele != 'thrpt']
    features = df[feature_keys].to_numpy()
    labels = df['thrpt'].to_numpy()
    return features, labels


def train_regression_autogluon(train_df, test_df):
    mx.npx.reset_np()
    from autogluon import TabularPrediction as task
    predictor = task.fit(train_data=task.Dataset(df=train_df),
                         output_directory=args.out_dir, label='thrpt',
                         eval_metric='mean_absolute_error')
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
    mx.npx.set_np()


def train_ranking_catboost(train_df, test_df):
    import catboost
    params = {'loss_function': 'YetiRank'}
    train_features, train_labels = get_feature_label(train_df)
    test_features, test_labels = get_feature_label(test_df)
    train_pool = catboost.Pool(data=train_features, label=train_labels)
    test_pool = catboost.Pool(data=test_features, label=test_labels)
    model = catboost.CatBoost(params)
    model.fit(X=train_pool)
    predict_result = model.predict(test_pool)
    logging.info(predict_result)


class PerfNet(gluon.HybridBlock):
    def __init__(self, num_hidden=64, num_layer=2, dropout=0.1, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.layers = nn.HybridSequential()
            with self.layers.name_scope():
                for i in range(num_layer):
                    self.layers.add(nn.Dense(num_hidden, flatten=False))
                    self.layers.add(nn.LeakyReLU(0.1))
                    self.layers.add(nn.Dropout(dropout))

    def hybrid_forward(self, F, data):
        return self.layers(data)


def get_nn_pred_scores(embed_net, regression_score_net, test_features,
                       feat_mean, feat_std, batch_size):
    pred_scores = []
    for i in range(0, test_features.shape[0], batch_size):
        batch_feature = (test_features[i:(i + batch_size)] - feat_mean) / feat_std
        embeddings = embed_net(batch_feature)
        scores = regression_score_net(embeddings)
        pred_scores.extend(scores.asnumpy().to_list())
    return pred_scores


def evaluate_nn(test_features, test_labels, embed_net, regression_score_net,
                rank_score_net, batch_size, num_hidden, ctx, threshold):
    n_samples = test_features.shape[0]
    embeddings = mx.np.zeros(shape=(n_samples, num_hidden), dtype=np.float32, ctx=ctx)
    for i in range(0, n_samples, batch_size):
        batch_features = mx.np.array(test_features[i:(i + batch_size)], dtype=np.float32,
                                     ctx=ctx)
        embeddings[i:(i + batch_size), :] = embed_net(batch_features)
    # Calculate ranking scores
    n_correct = 0
    n_total = 0
    total_nll = 0
    gt_label_distribution = np.zeros(shape=(3,), dtype=np.int64)
    for i in range(0, n_samples, batch_size):
        lhs_embeddings = embeddings[i:(i + batch_size)]
        lhs_labels = test_labels[i:(i + batch_size)]
        for j in range(0, n_samples, batch_size):
            rhs_embeddings = embeddings[j:(j + batch_size)]
            rhs_labels = test_labels[j:(j + batch_size)]
            pair_scores = np.expand_dims(lhs_labels, axis=1) - np.expand_dims(rhs_labels, axis=0)
            pair_label = np.zeros_like(pair_scores, dtype=np.int32)
            pair_label[pair_scores >= threshold] = 1
            pair_label[pair_scores <= -threshold] = 2
            pair_label = mx.np.array(pair_label, dtype=np.int32, ctx=ctx)
            inner_lhs_embeddings = mx.np.expand_dims(lhs_embeddings, axis=1)
            rhs_embeddings = mx.np.expand_dims(rhs_embeddings, axis=0)
            inner_lhs_embeddings, rhs_embeddings =\
                mx.np.broadcast_arrays(inner_lhs_embeddings, rhs_embeddings)
            joint_embedding = mx.np.concatenate([inner_lhs_embeddings, rhs_embeddings,
                                                 mx.np.abs(inner_lhs_embeddings - rhs_embeddings),
                                                 inner_lhs_embeddings * rhs_embeddings], axis=-1)
            pred_rank_label_scores = rank_score_net(joint_embedding)
            logits = mx.npx.pick(mx.npx.log_softmax(pred_rank_label_scores, axis=-1),
                                 pair_label)
            n_correct += (pred_rank_label_scores.argmax(axis=-1).astype(np.int32)
                          == pair_label).sum()
            n_total += np.prod(pair_label.shape)
            total_nll += -logits.sum()
            for k in range(3):
                gt_label_distribution[k] += (pair_label == k).sum()
    return total_nll / n_total, n_correct / n_total, n_correct, n_total, gt_label_distribution


def train_nn(args, train_df, test_df):
    ctx = parse_ctx(args.gpus)[0]
    batch_size = args.batch_size
    embed_net = PerfNet(args.num_hidden, args.num_layers, args.dropout)
    rank_score_net = nn.HybridSequential()
    with rank_score_net.name_scope():
        rank_score_net.add(nn.Dense(args.num_hidden, flatten=False))
        rank_score_net.add(nn.LeakyReLU(0.1))
        rank_score_net.add(nn.Dense(3, flatten=False))
    regression_score_net = nn.HybridSequential()
    with regression_score_net.name_scope():
        regression_score_net.add(nn.Dense(args.num_hidden, flatten=False))
        regression_score_net.add(nn.LeakyReLU(0.1))
        regression_score_net = nn.Dense(1, flatten=False)
    embed_net.hybridize()
    rank_score_net.hybridize()
    regression_score_net.hybridize()
    rank_loss_func = gluon.loss.SoftmaxCrossEntropyLoss(batch_axis=[0, 1])

    train_dev_features, train_dev_labels = get_feature_label(train_df)
    shuffle_idx = np.random.permutation(train_dev_features.shape[0])
    train_dev_features, train_dev_labels =\
        train_dev_features[shuffle_idx], train_dev_labels[shuffle_idx]
    num_train = train_dev_features.shape[0] - int(args.dev_ratio * train_dev_features.shape[0])
    train_features, train_labels = train_dev_features[:num_train], train_dev_labels[:num_train]
    dev_features, dev_labels = train_dev_features[num_train:], train_dev_labels[num_train:]
    test_features, test_labels = get_feature_label(test_df)
    embed_net.initialize(init=mx.init.Normal(0.01), ctx=ctx)
    rank_score_net.initialize(init=mx.init.Normal(0.01), ctx=ctx)
    regression_score_net.initialize(init=mx.init.Normal(0.01), ctx=ctx)
    optimizer_params = {'learning_rate': args.lr, 'wd': args.wd}
    embed_trainer = gluon.Trainer(embed_net.collect_params(), 'adam', optimizer_params)
    rank_score_trainer = gluon.Trainer(rank_score_net.collect_params(),
                                       'adam',
                                       optimizer_params)
    regression_score_trainer = gluon.Trainer(regression_score_net.collect_params(),
                                             'adam',
                                             optimizer_params)
    avg_regress_loss = 0
    avg_regress_loss_denom = 0
    avg_rank_loss = 0
    avg_rank_loss_denom = 0
    avg_embed_net_norm = 0
    avg_rank_score_net_norm = 0
    avg_regression_score_net_norm = 0
    avg_norm_iter = 0
    best_val_acc = 0
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
        rank_pair_feature = mx.np.array(rank_pair_feature, dtype=np.float32, ctx=ctx)
        pair_label = mx.np.array(pair_label, dtype=np.int32, ctx=ctx)
        with mx.autograd.record():
            lhs_embedding = embed_net(batch_feature)
            pred_score = regression_score_net(lhs_embedding)
            regress_loss = mx.np.abs(pred_score - batch_label).mean()

            rhs_embedding = embed_net(rank_pair_feature)
            # Concatenate the embedding
            lhs_embedding = mx.np.expand_dims(lhs_embedding, axis=1)
            lhs_embedding = mx.np.broadcast_to(lhs_embedding, rhs_embedding.shape)
            joint_embedding = mx.np.concatenate([lhs_embedding, rhs_embedding,
                                                 mx.np.abs(lhs_embedding - rhs_embedding),
                                                 lhs_embedding * rhs_embedding], axis=-1)
            rank_logits = rank_score_net(joint_embedding)
            rank_loss = rank_loss_func(rank_logits, pair_label).mean()
            loss = args.regress_alpha * regress_loss + args.rank_alpha * rank_loss
        loss.backward()
        embed_net_norm = grad_global_norm(embed_net.collect_params().values())
        rank_score_net_norm = grad_global_norm(rank_score_net.collect_params().values())
        regression_score_net_norm = grad_global_norm(regression_score_net.collect_params().values())
        avg_embed_net_norm += embed_net_norm
        avg_rank_score_net_norm += rank_score_net_norm
        avg_regression_score_net_norm += regression_score_net_norm
        avg_norm_iter += 1
        embed_trainer.step(1.0)
        rank_score_trainer.step(1.0)
        regression_score_trainer.step(1.0)
        avg_regress_loss += regress_loss.asnumpy() * batch_size
        avg_regress_loss_denom += batch_size
        avg_rank_loss += rank_loss.asnumpy() * batch_size * batch_size
        avg_rank_loss_denom += batch_size * batch_size
        if (i + 1) % args.log_interval == 0:
            logging.info('Iter:{}/{}, Train Loss Regression/Ranking={}/{}, '
                  'grad_norm Embed/Regression/Rank={}/{}/{}'
                  .format(i + 1, args.niter,
                          avg_regress_loss / avg_regress_loss_denom,
                          avg_rank_loss / avg_rank_loss_denom,
                          avg_embed_net_norm / avg_norm_iter,
                          avg_regression_score_net_norm / avg_norm_iter,
                          avg_rank_score_net_norm / avg_norm_iter))
            avg_regress_loss = 0
            avg_regress_loss_denom = 0
            avg_rank_loss = 0
            avg_rank_loss_denom = 0
            avg_embed_net_norm = 0
            avg_rank_score_net_norm = 0
            avg_regression_score_net_norm = 0
            avg_norm_iter = 0
        if (i + 1) % args.nval_iter == 0:
            val_nll, val_acc, n_correct, n_total, gt_label_distribution =\
                evaluate_nn(dev_features, dev_labels, embed_net, regression_score_net,
                            rank_score_net, batch_size, args.num_hidden, ctx, args.threshold)
            pair_label_total = gt_label_distribution.sum()
            logging.info('Validation error: nll={}, acc={},'
                         ' correct/total={}/{},'
                         ' dev distribution equal: {:.2f}, lhs>rhs: {:.2f}, lhs<rhs: {:.2f}'
                         .format(val_nll, val_acc, n_correct, n_total,
                                 gt_label_distribution[0] / pair_label_total * 100,
                                 gt_label_distribution[1] / pair_label_total * 100,
                                 gt_label_distribution[2] / pair_label_total * 100))
            if val_acc > best_val_acc:
                no_better_val_cnt = 0
                best_val_acc = val_acc
                embed_net.save_parameters(os.path.join(
                    args.out_dir, 'embed_net_best.params'))
                rank_score_net.save_parameters(os.path.join(
                    args.out_dir, 'rank_score_net_best.params'))
                best_val_loss_f.write('{}, {}, {}\n'.format(i + 1, best_val_acc, val_nll))
                test_nll, test_acc, test_n_correct, test_n_total, test_gt_label_distribution = \
                    evaluate_nn(test_features, test_labels, embed_net, regression_score_net,
                                rank_score_net, batch_size, args.num_hidden, ctx, args.threshold)
                test_loss_f.write('{}, {}, {}\n'.format(i + 1, test_acc, test_nll))
                logging.info('Test error: nll={}, acc={},'
                             ' correct/total={}/{},'
                             ' test distribution equal: {:.2f}, lhs>rhs: {:.2f}, lhs<rhs: {:.2f}'
                             .format(test_nll, test_acc, test_n_correct, test_n_total,
                                     test_gt_label_distribution[0] /
                                     test_gt_label_distribution.sum() * 100,
                                     test_gt_label_distribution[1] /
                                     test_gt_label_distribution.sum() * 100,
                                     test_gt_label_distribution[2] /
                                     test_gt_label_distribution.sum() * 100))
            else:
                no_better_val_cnt += 1
                if no_better_val_cnt > 5:
                    if curr_lr == 1E-5:
                        break
                    curr_lr = max(curr_lr / 2, 1E-5)
                    embed_trainer.set_learning_rate(curr_lr)
                    regression_score_trainer.set_learning_rate(curr_lr)
                    rank_score_trainer.set_learning_rate(curr_lr)
                    logging.info('Decrease learning rate to {}'.format(curr_lr))
                    no_better_val_cnt = 0
    best_val_loss_f.close()
    test_loss_f.close()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    logging_config(args.out_dir, 'thrpt_model')
    train_df, test_df = get_data(args)
    if args.algo == 'auto':
        train_regression_autogluon(train_df, test_df)
    elif args.algo == 'cat':
        train_ranking_catboost(train_df, test_df)
    elif args.algo == 'nn':
        train_nn(args, train_df, test_df)
    else:
        raise NotImplementedError
