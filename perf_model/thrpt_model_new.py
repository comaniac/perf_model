# pylint: disable=missing-docstring, invalid-name, ungrouped-imports
# pylint: disable=unsupported-assignment-operation, no-member
import argparse
import logging
import os
import glob
import multiprocessing
import json
import torch
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import random
try:
    import catboost
except Exception:
    import imp
import pandas as pd
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from sklearn.metrics import ndcg_score
from .util import analyze_valid_threshold, logging_config, read_pd
from .nn_ranker import RankGroupSampler, RankingModel, get_ranking_loss, RegressionSampler


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)


def down_sample_df(df, seed, ratio):
    """Down sampling data frame by a ratio based on operators.

    Parameters
    ----------
    df
        The input dataset in pandas DataFrame
    seed
        The seed to sample
    ratio
        The ratio of the remaining dataset

    Returns
    -------
    sampled_df
        The sampled dataframe
    """
    rng = np.random.RandomState(seed)

    op_keys = [k for k in df.columns.to_list() if k.find('in_') != -1 or k.find('attr_') != -1]
    if len(op_keys) > 0:
        group_dfs = [x for _, x in df.groupby(op_keys)]
    else:
        group_dfs = [df]
    sampled_dfs = [d.sample(int(len(d) * ratio), random_state=rng) for d in group_dfs]
    return pd.concat(sampled_dfs)


def split_df_by_op(df, seed, ratio):
    """Split the input data frame into Train + Test by operators.

    Parameters
    ----------
    df
        The input dataset in pandas DataFrame
    seed
        The seed to split the train + remaining
    ratio
        The ratio of the remaining dataset

    Returns
    -------
    train_df
        The training dataframe
    test_df
        The testing dataframe
    op_keys
    """
    rng = np.random.RandomState(seed)

    op_keys = [k for k in df.columns.to_list() if k.find('in_') != -1 or k.find('attr_') != -1]
    if len(op_keys) > 0:
        group_dfs = [x for _, x in df.groupby(op_keys)]
    else:
        group_dfs = [df]

    if len(group_dfs) == 1:
        return None, None
    print('ratio=', ratio)
    print(ratio * len(group_dfs))
    num = int(np.ceil(ratio * len(group_dfs)))
    perm = rng.permutation(len(group_dfs))
    train_num = len(group_dfs) - num
    print(f'Training Num: {train_num}, Test Num: {num}')
    train_dfs = [group_dfs[i] for i in perm[:train_num]]
    test_dfs = [group_dfs[i] for i in perm[train_num:]]
    return pd.concat(train_dfs), pd.concat(test_dfs), op_keys


def split_train_test_df(df, seed, ratio, top_sample_ratio=0.2, group_size=10, K=4):
    """Get the dataframe for testing

    Parameters
    ----------
    df
        The input DataFrame
    seed
        The seed
    ratio
        The ratio to split the testing set
    top_sample_ratio
        The ratio of putting top samples to the test set.
        This tests the ability of the model to predict the latency / ranking of
        the out-of-domain samples.
    group_size
        Num of samples in each group
    K
        For each sample in the test set.
        Sample K groups that contain this test sample.

    Returns
    -------
    train_df
        The training dataframe.
    test_df
        The testing dataframe. This contains samples in the test set,
        and can be used for regression analysis
    test_rank_group_all_tuple
        - test_rank_group_features_all
            (#samples, #group_size, #features)
        - test_rank_group_labels_all
            (#samples, #group_size)
    test_rank_group_valid_tuple
        - test_rank_group_features_valid
            (#samples, #group_size, #features)
        - test_rank_group_labels_valid
            (#samples, #group_size)
    """
    rng = np.random.RandomState(seed)
    num_samples = len(df)
    test_num = int(np.ceil(ratio * num_samples))
    train_num = len(df) - test_num
    thrpt = df['thrpt'].to_numpy()

    # Perform stratified sampling.
    # Here, we only consider two buckets: those with thrpt == 0 (invalid), and those that are valid.
    all_valid_indices = (thrpt > 0).nonzero()[0]
    all_invalid_indices = (thrpt == 0).nonzero()[0]
    rng.shuffle(all_valid_indices)
    rng.shuffle(all_invalid_indices)
    valid_test_num = int(np.ceil(len(all_valid_indices) * ratio))
    invalid_test_num = test_num - valid_test_num
    test_indices = np.concatenate([all_valid_indices[:valid_test_num],
                                   all_invalid_indices[:invalid_test_num]], axis=0)
    train_indices = np.concatenate([all_valid_indices[valid_test_num:],
                                    all_invalid_indices[invalid_test_num:]], axis=0)
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    # Get ranking dataframe, we sample two datasets:
    # 1) We sample the valid thrpts in the test set and draw other samples from the whole dataset
    # 2) We only sample groups from the valid thrpts.
    all_features, all_labels = get_feature_label(df)
    test_rank_arr = []
    for idx in all_valid_indices[:valid_test_num]:
        for _ in range(K):
            group_indices = (rng.choice(num_samples - 1,
                                        group_size - 1, True) + idx + 1) % num_samples
            group_indices = np.append(group_indices, idx)
            test_rank_arr.append(group_indices)
    # Shape (#samples, #group_size)
    test_rank_array = np.array(test_rank_arr, dtype=np.int64)
    # Shape (#samples, #group_size, #features)
    rank_group_features_all = np.take(all_features, test_rank_array, axis=0)
    # Shape (#samples, #group_size)
    rank_group_labels_all = np.take(all_labels, test_rank_array, axis=0)

    test_rank_arr = []
    for idx in all_valid_indices[:valid_test_num]:
        for _ in range(K):
            group_indices = (rng.choice(len(all_valid_indices) - 1,
                                        group_size - 1, True) + idx + 1) % len(all_valid_indices)
            group_indices = all_valid_indices[group_indices]
            group_indices = np.append(group_indices, idx)
            test_rank_arr.append(group_indices)
    test_rank_array = np.array(test_rank_arr, dtype=np.int64)
    rank_group_features_valid = np.take(all_features, test_rank_array, axis=0)
    rank_group_labels_valid = np.take(all_labels, test_rank_array, axis=0)
    return train_df, test_df,\
           (rank_group_features_all, rank_group_labels_all),\
           (rank_group_features_valid, rank_group_labels_valid)


def get_data(data_path, thrpt_threshold=0):
    """Load data from the data path

    Parameters
    ----------
    data_path
        The data path
    thrpt_threshold
        -1 --> adaptive valid thrpt
        0 --> no adaptive valid thrpt

    Returns
    -------
    df
        The DataFrame
    """
    if thrpt_threshold == -1:
        invalid_thd = analyze_valid_threshold(data_path)
    else:
        invalid_thd = 0
    df = read_pd(data_path)
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
    return df, used_keys


def get_feature_label(df):
    feature_keys = [ele for ele in df.keys() if ele != 'thrpt']
    features = df[feature_keys].to_numpy()
    labels = df['thrpt'].to_numpy()
    return features, labels


def get_group_indices(df):
    op_keys = [k for k in df.columns.to_list() if k.find('in_') != -1
               or k.find('attr_') != -1]
    if len(op_keys) > 0:
        group_indices = [idx for _, idx in df.groupby(op_keys).groups.items()]
    else:
        group_indices = np.arange(len(df))
    return group_indices


def get_group_df(df):
    op_keys = [k for k in df.columns.to_list() if k.find('in_') != -1 or k.find('attr_') != -1]
    if len(op_keys) > 0:
        group_dfs = [x for _, x in df.groupby(op_keys)]
    else:
        group_dfs = [df]
    return group_dfs


def group_ndcg_score(truth, prediction, k=None, group_indices=None):
    if group_indices is None:
        return ndcg_score(np.expand_dims(truth, axis=0),
                          np.expand_dims(prediction, axis=0), k=k)
    else:
        avg_ndcg = 0
        cnt = 0
        for sel in group_indices:
            sel_truth = truth[sel]
            sel_prediction = prediction[sel]
            if len(sel) == 1:
                continue
            else:
                try:
                    group_ndcg = ndcg_score(np.expand_dims(sel_truth, axis=0),
                                            np.expand_dims(sel_prediction, axis=0), k=k)
                    avg_ndcg += group_ndcg
                    cnt += 1
                except Exception:
                    print(sel_truth)
                    print(sel_prediction)
                    raise Exception
        avg_ndcg /= cnt
        return avg_ndcg


class CatRegressor:
    def __init__(self, model=None):
        self.model = model

    def fit(self, train_df, valid_df=None, train_dir='.', niter=10000, seed=123):
        if self.model is None:
            params = {
                'loss_function': 'RMSE',
                'task_type': 'GPU',
                'iterations': niter,
                'verbose': True,
                'train_dir': train_dir,
                'random_seed': seed
            }
            self.model = catboost.CatBoost(params)
            init_model = None
        else:
            init_model = self.model
        train_features, train_labels = get_feature_label(train_df)
        train_pool = catboost.Pool(data=train_features,
                                   label=train_labels)
        if valid_df is not None:
            valid_features, valid_labels = get_feature_label(valid_df)
            dev_pool = catboost.Pool(data=valid_features,
                                     label=valid_labels)
        else:
            dev_pool = None
        self.model.fit(train_pool, eval_set=dev_pool, init_model=init_model)

    @classmethod
    def load(cls, path):
        try:
            model = catboost.CatBoost().load_model(os.path.join(path, 'cat_regression.cbm'))
            return cls(model=model)
        except NameError:  # CatBoost is unavailable. Try to load Python model.
            pass

    def save(self, out_dir):
        self.model.save_model(os.path.join(out_dir, 'cat_regression.cbm'))
        self.model.save_model(os.path.join(out_dir, 'cat_regression'), format='python')

    def predict(self, features):
        features = np.array(features, dtype=np.float32)
        features_shape = features.shape
        # features = features.reshape((-1, features_shape[-1]))
        # with multiprocessing.Pool(max(os.cpu_count() - 1, 1)) as pool:
        #     preds = pool.map(self.model.predict, features)
        # preds = np.array(preds)
        preds = self.model.predict(features.reshape((-1, features_shape[-1])))
        preds = preds.reshape(features_shape[:-1])
        return preds

    def evaluate(self, features, labels, mode='regression'):
        preds = self.predict(features)
        if mode == 'regression':
            original_preds = preds
            preds = preds * (preds > 0)
            rmse = np.sqrt(np.mean(np.square(preds - labels)))
            mae = np.mean(np.abs(preds - labels))
            valid_indices = (labels > 0).nonzero()[0]
            valid_rmse = np.sqrt(np.mean(np.square(preds[valid_indices] - labels[valid_indices])))
            valid_mae = np.mean(np.abs(preds[valid_indices] - labels[valid_indices]))
            return {'rmse': rmse, 'mae': mae,
                    'valid_rmse': valid_rmse, 'valid_mae': valid_mae,
                    'invalid_acc': np.sum((original_preds <= 0) * (labels <= 0)) / len(preds)}
        elif mode == 'ranking':
            # We calculate two things, the NDCG score and the MRR score.
            ndcg_val = ndcg_score(y_true=labels, y_score=preds)
            ndcg_K3_val = ndcg_score(y_true=labels, y_score=preds, k=3)
            abs_ndcg_score = ndcg_score(y_true=np.argsort(labels), y_score=preds)
            abs_ndcg_k3_score = ndcg_score(y_true=np.argsort(labels),
                                           y_score=preds,
                                           k=3)
            rel_ndcg_score = ndcg_score(y_true=labels / (labels.max(axis=-1, keepdims=True) + 1E-6),
                                        y_score=preds)
            rel_ndcg_k3_score = ndcg_score(y_true=labels / (labels.max(axis=-1, keepdims=True)
                                                            + 1E-6),
                                           y_score=preds,
                                           k=3)
            ranks = np.argsort(-preds, axis=-1) + 1
            true_max_indices = np.argmax(labels, axis=-1)
            rank_of_max = ranks[np.arange(len(true_max_indices)), true_max_indices]
            mrr = np.mean(1.0 / rank_of_max)
            return {'ndcg': ndcg_val,
                    'ndcg_k3': ndcg_K3_val,
                    'abs_ndcg': abs_ndcg_score,
                    'abs_ndcg_k3': abs_ndcg_k3_score,
                    'rel_ndcg': rel_ndcg_score,
                    'rel_ndcg_k3': rel_ndcg_k3_score,
                    'mrr': mrr, 'rank_of_top': 1 / mrr}
        else:
            raise NotImplementedError


class CatBoostPoolIndicesGenerator:
    def __init__(self, thrpt, sample_num=10240, group_size=20, method='random'):
        self.thrpt = thrpt
        self.total = len(thrpt)
        self.sample_num = sample_num
        self.group_size = min(group_size, self.total // 2)
        self.generator = np.random.default_rng()

    def __call__(self):
        out = []
        for i in range(self.sample_num):
            out.append(self.generator.choice(self.total, self.group_size, False))
        indices = np.vstack(out)
        return indices


class CatRanker(CatRegressor):
    def __init__(self, model=None, normalize_relevance=False):
        self.model = model
        self.normalize_relevance = normalize_relevance

    def fit(self, train_df, step_sample_num=204800, group_size=40,
            niter=5000, train_dir='.', seed=123):
        if self.model is not None:
            init_model = self.model
        else:
            init_model = None
        params = {
            'loss_function': 'YetiRank',
            'task_type': 'GPU',
            'iterations': niter,
            'verbose': True,
            'train_dir': train_dir,
            'random_seed': seed
        }
        num_fit_calls = 1
        step_sample_num = min(step_sample_num, len(train_df) * 5)
        self.model = catboost.CatBoost(params)
        features, thrpt = get_feature_label(train_df)
        sampler = CatBoostPoolIndicesGenerator(thrpt,
                                               sample_num=step_sample_num,
                                               group_size=group_size)
        for i in range(num_fit_calls):
            indices = sampler()
            step_features = np.take(features, indices, axis=0)
            step_thrpt = np.take(thrpt, indices, axis=0)
            if self.normalize_relevance:
                step_thrpt = step_thrpt / (np.max(step_thrpt, axis=-1, keepdims=True) + 1E-6)
            step_groups = np.broadcast_to(np.arange(step_thrpt.shape[0]).reshape((-1, 1)),
                                          step_thrpt.shape)
            train_pool = catboost.Pool(data=step_features.reshape((-1, step_features.shape[-1])),
                                       label=step_thrpt.reshape((-1,)),
                                       group_id=step_groups.reshape((-1,)))
            self.model.fit(train_pool, init_model=init_model)

    def save(self, out_dir):
        self.model.save_model(os.path.join(out_dir, 'cat_ranking.cbm'))
        self.model.save_model(os.path.join(out_dir, 'cat_ranking'), format='python')

    @classmethod
    def load(cls, path):
        try:
            model = catboost.CatBoost().load_model(os.path.join(path, 'cat_ranking.cbm'))
            return cls(model=model)
        except NameError:  # CatBoost is unavailable. Try to load Python model.
            pass


class NNRanker:
    def __init__(self, in_units=None, units=512, num_layers=3,
                 dropout=0.1, act_type='leaky',
                 rank_loss_fn='lambda_rank',
                 beta_distribution=(3.0, 1.0),
                 neg_mult=5, use_gate=True,
                 mean_val=None, std_val=None):
        if in_units is None:
            self.net = None
        else:
            self.net = RankingModel(in_units=in_units,
                                    units=units,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    use_gate=use_gate,
                                    act_type=act_type)
        self._in_units = in_units
        self._units = units
        self._use_gate = use_gate
        self._num_layers = num_layers
        self._dropout = dropout
        self._act_type = act_type
        self._rank_loss_fn = rank_loss_fn
        self._beta_distribution = beta_distribution
        self._neg_mult = neg_mult
        self._mean_val = mean_val
        self._std_val = std_val

    def fit(self, train_df, batch_size=256, group_size=10, lr=1E-2,
            iter_mult=500, rank_lambda=1.0, test_df=None, train_dir='.'):
        split_ratio = 0.05
        train_df, valid_df, _ = split_df_by_op(train_df, seed=100, ratio=split_ratio)
        if test_df is not None:
            test_group_indices = get_group_indices(test_df)
        # features, labels = get_feature_label(train_df)
        # logging.info(f'#Train = {len(train_df)},'
        #              f' #Non-invalid Throughputs in Train = {len((labels >0).nonzero()[0])}')
        # train_num = int(np.ceil((1 - split_ratio) * len(features)))
        # perm = np.random.permutation(len(features))
        # train_features, train_labels = features[perm[:train_num]], labels[perm[:train_num]]
        # valid_features, valid_labels = features[perm[train_num:]], labels[perm[train_num:]]
        features, labels = get_feature_label(train_df)
        valid_features, valid_labels = get_feature_label(valid_df)

        if test_df is not None:
            test_features, test_labels = get_feature_label(test_df)
        epoch_iters = (len(features) + batch_size - 1) // batch_size
        log_interval = epoch_iters * iter_mult // 500
        num_iters = epoch_iters * iter_mult
        if self.net is None:
            self._in_units = features.shape[1]
            self.net = RankingModel(in_units=features.shape[1],
                                    units=self._units,
                                    num_layers=self._num_layers,
                                    dropout=self._dropout,
                                    use_gate=self._use_gate,
                                    act_type=self._act_type)
        self.net.cuda()
        self.net.train()
        non_invalid_labels = labels[labels > 0]
        if self._mean_val is None:
            mean_val = non_invalid_labels.mean()
            std_val = non_invalid_labels.std()
            self._mean_val = mean_val
            self._std_val = std_val
        else:
            mean_val = self._mean_val
            std_val = self._std_val
        th_features = th.tensor(features, dtype=th.float32)
        th_labels = th.tensor(labels, dtype=th.float32)
        dataset = TensorDataset(th_features, th_labels)
        if self._rank_loss_fn == 'no_rank':
            batch_sampler = RegressionSampler(thrpt=labels,
                                              regression_batch_size=batch_size * group_size)
        else:
            batch_sampler = RankGroupSampler(thrpt=labels,
                                             rank_batch_size=batch_size,
                                             group_size=group_size,
                                             beta_params=self._beta_distribution)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                  T_max=num_iters,
                                                                  eta_min=1E-5)
        if self._rank_loss_fn != 'no_rank':
            rank_loss_fn = get_ranking_loss(self._rank_loss_fn)
        dataloader = iter(dataloader)
        log_regression_loss = 0
        log_ranking_loss = 0
        log_cnt = 0
        niter = 0
        epoch_iter = 0
        best_valid_rmse = np.inf
        no_better = 0
        stop_patience = 50
        for ranking_features, ranking_labels in dataloader:
            optimizer.zero_grad()
            ranking_features = ranking_features.cuda()
            ranking_labels = ranking_labels.cuda()
            if self._rank_loss_fn != 'no_rank':
                ranking_labels = ranking_labels.reshape((batch_size, group_size))
                original_ranking_labels = ranking_labels
                ranking_labels = (ranking_labels - mean_val) / std_val
                ranking_scores = self.net(ranking_features)
                ranking_scores = ranking_scores.reshape((batch_size, group_size))
                loss_regression = torch.square(ranking_scores - ranking_labels).mean()
                loss_ranking = rank_loss_fn(y_pred=ranking_scores,
                                            y_true=original_ranking_labels / std_val)
                loss = loss_regression + rank_lambda * loss_ranking
            else:
                ranking_labels = (ranking_labels - mean_val) / std_val
                ranking_scores = self.net(ranking_features)
                loss_regression = torch.square(ranking_scores - ranking_labels).mean()
                loss = loss_regression
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                log_regression_loss += loss_regression
                if self._rank_loss_fn != 'no_rank':
                    log_ranking_loss += loss_ranking
                log_cnt += 1
                if log_cnt >= log_interval:
                    logging.info('[{}/{}] Regression Loss = {:.4f}, Ranking Loss = {:.4f}'
                                 .format(niter + 1, num_iters,
                                         log_regression_loss / log_cnt,
                                         log_ranking_loss / log_cnt))
                    log_regression_loss = 0
                    log_ranking_loss = 0
                    log_cnt = 0
                    valid_score = self.evaluate(valid_features, valid_labels, 'regression')
                    logging.info(f'[{niter + 1}/{num_iters}], Valid_score={valid_score}')
                    if valid_score['rmse'] < best_valid_rmse:
                        best_valid_rmse = valid_score['rmse']
                        torch.save(self.net.state_dict(), os.path.join(train_dir,
                                                                       'best_model_states.th'))
                        no_better = 0
                    else:
                        no_better += 1
                    if test_df is not None:
                        test_score = self.evaluate(test_features, test_labels, 'regression',
                                                   group_indices=test_group_indices)
                        logging.info(f'[{niter + 1}/{num_iters}], Test_score={test_score}')
            niter += 1
            epoch_iter += 1
            if epoch_iter >= epoch_iters:
                epoch_iter = 0
            if niter >= num_iters:
                break
            if no_better >= stop_patience:
                logging.info('Early stop')
                break
        self.net.load_state_dict(torch.load(os.path.join(train_dir, 'best_model_states.th')))

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(out_dir, 'model_states.th'))
        model_cfg = {'in_units': self._in_units,
                     'units': self._units,
                     'num_layers': self._num_layers,
                     'dropout': self._dropout,
                     'act_type': self._act_type,
                     'use_gate': self._use_gate,
                     'rank_loss_fn': self._rank_loss_fn,
                     'beta_distribution': self._beta_distribution,
                     'neg_mult': self._neg_mult,
                     'mean_val': self._mean_val,
                     'std_val': self._std_val}
        with open(os.path.join(out_dir, 'model_config.json'), 'w') as out_f:
            json.dump(model_cfg, out_f)

    @classmethod
    def load(cls, model_dir):
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as in_f:
            cfg = json.load(in_f)
        model = cls(**cfg)
        if torch.cuda.is_available() is False:
            model.net.load_state_dict(torch.load(os.path.join(model_dir, 'model_states.th'),
                                                 map_location=torch.device('cpu')))
        else:
            model.net.load_state_dict(torch.load(os.path.join(model_dir, 'model_states.th')))
        return model

    def predict(self, features, use_gpu=False):
        features = np.array(features, dtype=np.float32)
        if use_gpu is False:
            self.net.cpu()
        else:
            self.net.cuda()
        self.net.eval()
        batch_size = 10240
        with torch.no_grad():
            all_preds = []
            for batch_start in range(0, features.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, features.shape[0])
                th_features = torch.tensor(features[batch_start:batch_end], dtype=th.float32)
                if use_gpu:
                    th_features = th_features.cuda()
                features_shape = th_features.shape
                preds = self.net(th_features.reshape((-1, features_shape[-1])))
                preds = preds.reshape(features_shape[:-1]) * self._std_val + self._mean_val
                if use_gpu:
                    preds = preds.cpu()
                preds = preds.numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)

    def evaluate(self, features, labels, mode='ranking', group_indices=None):
        preds = self.predict(features, use_gpu=True)
        if mode == 'regression':
            original_preds = preds
            preds = preds * (preds > 0)
            rmse = np.sqrt(np.mean(np.square(preds - labels)))
            mae = np.mean(np.abs(preds - labels))
            valid_indices = (labels > 0).nonzero()[0]
            valid_rmse = np.sqrt(np.mean(np.square(preds[valid_indices] - labels[valid_indices])))
            valid_mae = np.mean(np.abs(preds[valid_indices] - labels[valid_indices]))
            ndcg_top_2 = ndcg_score(np.expand_dims(labels, axis=0),
                                    np.expand_dims(original_preds, axis=0), k=2)
            ndcg_top_8 = ndcg_score(np.expand_dims(labels, axis=0),
                                    np.expand_dims(original_preds, axis=0), k=8)
            group_ndcg_top2 = group_ndcg_score(labels, original_preds, k=2,
                                               group_indices=group_indices)
            group_ndcg_top8 = group_ndcg_score(labels, original_preds, k=8,
                                               group_indices=group_indices)
            return {'rmse': rmse, 'mae': mae,
                    'valid_rmse': valid_rmse, 'valid_mae': valid_mae,
                    'invalid_acc': np.sum((original_preds <= 0) * (labels <= 0)) / len(preds),
                    'ndcg_top_2': ndcg_top_2,
                    'ndcg_top_8': ndcg_top_8,
                    'group_ndcg_top2': group_ndcg_top2,
                    'group_ndcg_top8': group_ndcg_top8}
        elif mode == 'ranking':
            # We calculate two things, the NDCG score and the MRR score.
            ndcg_val = ndcg_score(y_true=labels, y_score=preds)
            ndcg_K3_val = ndcg_score(y_true=labels, y_score=preds, k=3)
            abs_ndcg_score = ndcg_score(y_true=np.argsort(labels),
                                        y_score=preds)
            abs_ndcg_k3_score = ndcg_score(y_true=np.argsort(labels),
                                           y_score=preds, k=3)
            ranks = np.argsort(-preds, axis=-1) + 1
            true_max_indices = np.argmax(labels, axis=-1)
            rank_of_max = ranks[np.arange(len(true_max_indices)), true_max_indices]
            mrr = np.mean(1.0 / rank_of_max)
            return {'ndcg': ndcg_val,
                    'ndcg_k3': ndcg_K3_val,
                    'abs_ndcg': abs_ndcg_score,
                    'abs_ndcg_k3_score': abs_ndcg_k3_score,
                    'mrr': mrr, 'rank_of_top': 1 / mrr}
        else:
            raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(description='Performance Model')
    parser.add_argument('--seed',
                        type=int,
                        default=100,
                        help='Seed for the training.')
    parser.add_argument('--data_prefix',
                        type=str,
                        default=None,
                        help='Prefix of the training/validation/testing dataset.')
    parser.add_argument('--out_dir',
                        type=str,
                        default='thrpt_model_out',
                        help='output path of the throughput model.')
    parser.add_argument('--split_test', action='store_true',
                        help='When turned on, we will try to split the data into training, '
                             'and testing.')
    parser.add_argument('--subsample', action='store_true',
                        help='When set, it will subsample the input training df.')
    parser.add_argument('--subsample_ratio', default=0.7, type=float,
                        help='The ratio of the subsample')
    parser.add_argument('--split_test_op_level', action='store_true')
    split_args = parser.add_argument_group('data split arguments')
    split_args.add_argument('--dataset',
                            type=str,
                            default=None,
                            help='path to the input csv file.')
    split_args.add_argument('--save_used_keys', action='store_true',
                            help='Store the used keys.')
    split_args.add_argument('--used_key_path', default=None,
                            help='Path of the used key.')
    split_args.add_argument('--split_train_name', default=None,
                            help='Name of the training split.')
    split_args.add_argument('--split_test_name', default=None,
                            help='Name of the testing split.')
    split_args.add_argument('--split_rank_test_prefix', default=None,
                            help='Prefix of the rank test datasets.')
    split_args.add_argument('--split_test_ratio', default=0.1, type=float,
                            help='Ratio of the test set in the split.')
    split_args.add_argument('--split_top_ratio', default=0.0,
                            help='Ratio of the top samples that will be split to the test set.')
    split_args.add_argument('--split_rank_group_size', default=10,
                            help='Size of each rank group.')
    split_args.add_argument('--split_rank_K', default=20,
                            help='K of each rank group.')
    parser.add_argument('--algo',
                        choices=['cat_regression',
                                 'cat_ranking',
                                 'nn'],
                        default='cat_regression',
                        help='The algorithm to use.')
    parser.add_argument('--iter_mult', default=500, type=int,
                        help='Lambda value of the ranking loss.')
    parser.add_argument('--rank_lambda', default=1.0, type=float,
                        help='Lambda value of the ranking loss.')
    parser.add_argument('--beta', default='3.0,1.0', type=str,
                        help='Beta distribution of the pos + neg samples.')
    parser.add_argument('--neg_mult', default=5, type=int,
                        help='The multiplier of #negative samples v.s. #positive samples.')
    parser.add_argument('--rank_loss_type', choices=['no_rank',
                                                     'lambda_rank',
                                                     'lambda_rank_hinge',
                                                     'approx_ndcg'],
                        default='lambda_rank_hinge',
                        help='Rank loss type.')
    parser.add_argument('--split_postfix', default=None, type=str,
                        help='split postfix')
    parser.add_argument('--batch_size', default=2560, type=int,
                        help='Batch size of the input.')
    parser.add_argument('--hidden_size', default=512, type=int,
                        help='Batch size of the input.')
    parser.add_argument('--num_layers', default=3, type=int,
                        help='Number of layers.')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout ratio.')
    parser.add_argument('--act_type', default='leaky', type=str,
                        help='Activation type.')
    parser.add_argument('--use_gate', default=1, type=int,
                        help='Whether to use gated network.')
    parser.add_argument('--niter', type=int, default=5000,
                        help='Number of iterations to train the catboost models.')
    parser.add_argument('--normalize_relevance', action='store_true',
                        help='Whether to turn on normalized relevance in ranking')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.split_test:
        logging_config(args.out_dir, 'split_data')
        df, used_keys = get_data(args.dataset)
        train_df, test_df, test_rank_group_sample_all, test_rank_group_sample_valid =\
            split_train_test_df(df,
                                args.seed,
                                args.split_test_ratio,
                                args.split_top_ratio,
                                args.split_rank_group_size,
                                args.split_rank_K)
        logging.info('Generate train data to {}, test data to {}, test rank data to {}'
                     .format(args.split_train_name,
                             args.split_test_name,
                             args.split_rank_test_prefix))
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        train_df.to_parquet(args.split_train_name)
        test_df.to_parquet(args.split_test_name)
        np.savez(args.split_rank_test_prefix + '.all.npz',
                 rank_features=test_rank_group_sample_all[0],
                 rank_labels=test_rank_group_sample_all[1])
        np.savez(args.split_rank_test_prefix + '.valid.npz',
                 rank_features=test_rank_group_sample_valid[0],
                 rank_labels=test_rank_group_sample_valid[1])
        logging.info('  #Train = {}, #Test = {}, #Ranking Test Groups = {}'
                     .format(len(train_df),
                             len(test_df),
                             len(test_rank_group_sample_all[0])))
        if args.save_used_keys:
            with open(args.used_key_path, 'w') as of:
                json.dump(used_keys, of)
    elif args.split_test_op_level:
        df, used_keys = get_data(args.dataset)
        train_df, test_df, op_keys = split_df_by_op(df, seed=args.seed, ratio=args.split_test_ratio)
        if train_df is None:
            logging.info(f'Cannot split {args.dataset}.')
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        train_df.to_parquet(args.split_train_name)
        test_df.to_parquet(args.split_test_name)
        logging.info('  #Train = {}, #Test = {}'.format(len(train_df), len(test_df)))
        if args.save_used_keys:
            with open(args.used_key_path, 'w') as of:
                json.dump(used_keys, of)
    elif args.subsample:
        sample_counts = []
        os.makedirs(args.out_dir, exist_ok=True)
        for folder in sorted(os.listdir(args.dataset)):
            if not os.path.isdir(os.path.join(args.dataset, folder)):
                continue
            os.makedirs(os.path.join(args.out_dir, folder), exist_ok=True)
            for data_path in sorted(glob.glob(os.path.join(args.dataset, folder, '*.train.pq'))):
                base_name = os.path.basename(data_path)[:-len('.train.pq')]
                if base_name in ['dense_small_batch.cuda',
                                 'conv2d_cudnn.cuda',
                                 'dense_cublas.cuda',
                                 'dense_large_batch.cuda']:
                    continue
                test_path = os.path.join(args.dataset, folder, base_name + '.test.pq')
                df = pd.read_parquet(data_path)
                test_df = pd.read_parquet(test_path)
                sample_counts.append((os.path.join(folder, base_name), len(df), len(test_df)))
                sub_df = down_sample_df(df, seed=args.seed, ratio=args.subsample_ratio)
                sub_df.to_parquet(os.path.join(args.out_dir, folder, base_name + '.train.pq'))
        df = pd.DataFrame(sample_counts)
        df.to_csv(os.path.join(args.out_dir, 'status.csv'))
    else:
        logging_config(args.out_dir, 'train')
        if args.split_postfix is not None and args.split_postfix != '1':
            if 'split_tuning_dataset_op' in args.data_prefix:
                real_prefix = args.data_prefix.replace('split_tuning_dataset_op',
                                                       f'split_tuning_dataset_op_{args.split_postfix}')
            elif 'split_tuning_dataset' in args.data_prefix:
                real_prefix = args.data_prefix.replace('split_tuning_dataset',
                                                       f'split_tuning_dataset_{args.split_postfix}')
            else:
                raise NotImplementedError
            train_df = read_pd(real_prefix + '.train.pq')
        else:
            train_df = read_pd(args.data_prefix + '.train.pq')
        test_df = read_pd(args.data_prefix + '.test.pq')
        # rank_test_all = np.load(args.data_prefix + '.rank_test.all.npz')
        # rank_test_valid = np.load(args.data_prefix + '.rank_test.valid.npz')
        with open(args.data_prefix + '.used_key.json', 'r') as in_f:
            used_key = json.load(in_f)
        train_df = train_df[used_key]
        test_df = test_df[used_key]
        if args.algo == 'cat_regression':
            model = CatRegressor()
            model.fit(train_df, valid_df=None, train_dir=args.out_dir, seed=args.seed,
                      niter=args.niter)
            model.save(args.out_dir)
            test_features, test_labels = get_feature_label(test_df)
            test_score = model.evaluate(test_features, test_labels, 'regression')
            # test_ranking_score_all = model.evaluate(rank_test_all['rank_features'],
            #                                         rank_test_all['rank_labels'],
            #                                         'ranking')
            # test_ranking_score_all = {k + '_all': v for k, v in test_ranking_score_all.items()}
            # test_ranking_score_valid = model.evaluate(rank_test_valid['rank_features'],
            #                                           rank_test_valid['rank_labels'],
            #                                           'ranking')
            # test_ranking_score_valid = {k + '_valid': v for k, v in test_ranking_score_valid.items()}
            # test_score.update(test_ranking_score_all)
            # test_score.update(test_ranking_score_valid)
            logging.info('Test Score={}'.format(test_score))
            with open(os.path.join(args.out_dir, 'test_scores.json'), 'w') as out_f:
                json.dump(test_score, out_f)
        elif args.algo == 'cat_ranking':
            model = CatRanker(normalize_relevance=args.normalize_relevance)
            model.fit(train_df, train_dir=args.out_dir, seed=args.seed, niter=args.niter)
            model.save(args.out_dir)
            test_score = {}
            # test_ranking_score_all = model.evaluate(rank_test_all['rank_features'],
            #                                         rank_test_all['rank_labels'],
            #                                         'ranking')
            # test_ranking_score_all = {k + '_all': v for k, v in test_ranking_score_all.items()}
            # test_ranking_score_valid = model.evaluate(rank_test_valid['rank_features'],
            #                                           rank_test_valid['rank_labels'],
            #                                           'ranking')
            # test_ranking_score_valid = {k + '_valid': v for k, v in
            #                             test_ranking_score_valid.items()}
            # test_score.update(test_ranking_score_all)
            # test_score.update(test_ranking_score_valid)
            logging.info('Test Score={}'.format(test_score))
            with open(os.path.join(args.out_dir, 'test_scores.json'), 'w') as out_f:
                json.dump(test_score, out_f)
        elif args.algo == 'nn':
            beta_distribution = [float(ele) for ele in args.beta.split(',')]
            model = NNRanker(units=args.hidden_size,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                             act_type=args.act_type,
                             use_gate=args.use_gate,
                             rank_loss_fn=args.rank_loss_type,
                             beta_distribution=beta_distribution,
                             neg_mult=args.neg_mult)
            model.fit(train_df,
                      rank_lambda=args.rank_lambda,
                      iter_mult=args.iter_mult,
                      test_df=test_df,
                      train_dir=args.out_dir)
            model.save(args.out_dir)
            test_group_indices = get_group_indices(test_df)
            test_features, test_labels = get_feature_label(test_df)
            test_score = model.evaluate(test_features, test_labels, 'regression',
                                        group_indices=test_group_indices)
            # test_ranking_score_all = model.evaluate(rank_test_all['rank_features'],
            #                                         rank_test_all['rank_labels'],
            #                                         'ranking')
            # test_ranking_score_all = {k + '_all': v for k, v in test_ranking_score_all.items()}
            # test_score.update(test_ranking_score_all)
            # test_ranking_score_valid = model.evaluate(rank_test_valid['rank_features'],
            #                                           rank_test_valid['rank_labels'],
            #                                           'ranking')
            # test_ranking_score_valid = {k + '_valid': v for k, v in
            #                             test_ranking_score_valid.items()}
            # test_score.update(test_ranking_score_valid)
            logging.info('Test Score={}'.format(test_score))
            with open(os.path.join(args.out_dir, 'test_scores.json'), 'w') as out_f:
                json.dump(test_score, out_f)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
