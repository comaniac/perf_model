"""Use BERT as the performance model."""
# pylint: disable=invalid-name
import argparse
import os
import sys

import catboost
import numpy as np
import pandas as pd
import tqdm

import logger
from util import analyze_valid_threshold


log = logger.get_logger('VALID')

def get_data(args):
    invalid_thd = analyze_valid_threshold(args.dataset)

    df = pd.read_csv(args.dataset)
    log.info('Invalid throughput is set to %.1f GFLOP/s', invalid_thd)
    df['thrpt'] = df['thrpt'].apply(lambda t: 1 if t >= invalid_thd else 0)
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
        else:
            used_keys.append(key)
    log.info('Original keys=%s, Not used keys=%s', list(df.keys()), not_used_keys)
    df = df[used_keys]
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


def train_model(args, reporter=None):
    """Training process."""
    logger.enable_log_file('valid.log')

    params = {'learning_rate': 1,
              'task_type': 'GPU',
              'iterations': args.n_iter,
              'verbose': False,
              'train_dir': args.out_dir,
              'random_seed': 17,
              'loss_function': 'CrossEntropy'}

    log.info('Loading data from %s...', args.dataset)
    train_df, test_df = get_data(args)
    train_dev_features, train_dev_labels = get_feature_label(train_df)
    test_features, test_labels = get_feature_label(test_df)

    # Split Train/Dev
    shuffle_idx = np.random.permutation(train_dev_features.shape[0])
    train_dev_features, train_dev_labels = \
        train_dev_features[shuffle_idx], train_dev_labels[shuffle_idx]
    num_train = train_dev_features.shape[0] - int(args.dev_ratio * train_dev_features.shape[0])
    train_features, train_labels = train_dev_features[:num_train], train_dev_labels[:num_train]
    dev_features, dev_labels = train_dev_features[num_train:], train_dev_labels[num_train:]

    log.info('Training...')
    train_pool = catboost.Pool(data=train_features, label=train_labels)
    dev_pool = catboost.Pool(data=dev_features, label=dev_labels)
    model = catboost.CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=dev_pool)

    log.info('Testing...')
    pred = model.predict(test_features)
    acc = 100.0 * sum([p == a for p, a in zip(pred, test_labels)]) / len(test_labels)
    log.info('Accuracy %.2f%%', acc)
    model.save_model('{}/valid_model.cbm'.format(args.out_dir))

def main():
    main_args = argparse.Namespace()
    main_args.dataset = sys.argv[1]
    main_args.n_iter = 500
    main_args.dev_ratio = 0.1
    main_args.test_ratio = 0.1
    main_args.out_dir = sys.argv[2]
    train_model(main_args)


if __name__ == "__main__":
    main()

