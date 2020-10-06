import pandas as pd
import argparse
import os
import json
import numpy as np
from perf_model.thrpt_model_new import NNRanker, read_pd, get_feature_label, CatRegressor,\
    CatRanker, get_group_indices
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import ndcg_score


parser = argparse.ArgumentParser(description='Read model results.')
parser.add_argument('--dir_path', type=str, default='model_results/nn_5.0_200')
parser.add_argument('--model_type', choices=['nn', 'cat_regression', 'cat_ranking'])
parser.add_argument('--eval_correlation', action='store_true')
parser.add_argument('--ranking_only', action='store_true')
parser.add_argument('--use_op_split', action='store_true')
parser.add_argument('--correlation_out_name', default=None, type=str)
args = parser.parse_args()


def group_ndcg_score(truth, prediction, k=None, group_indices=None):
    if group_indices is None:
        return ndcg_score(truth, prediction, k=k)
    else:
        avg_ndcg = 0
        cnt = 0
        for sel in group_indices:
            sel_truth = truth[sel]
            sel_prediction = prediction[sel]
            if len(sel) == 1:
                pass
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


correlation_dat = []
for dir_name in sorted(os.listdir(args.dir_path)):
    if not os.path.isdir(os.path.join(args.dir_path, dir_name)):
        continue
    for exp_name in sorted(os.listdir(os.path.join(args.dir_path, dir_name))):
        if exp_name in ['dense_small_batch.cuda', 'conv2d_cudnn.cuda',
                         'dense_cublas.cuda', 'dense_large_batch.cuda',
                         'conv2d_transpose_nchw.cuda',
                         'dense_tensorcore.cuda']:
            continue
        if args.use_op_split:
            if exp_name in ['dense_pack.x86']:
                continue
        if not os.path.isdir(os.path.join(args.dir_path, dir_name, exp_name)):
            continue
        if args.model_type == 'nn':
            model = NNRanker.load(os.path.join(args.dir_path, dir_name, exp_name))
        elif args.model_type == 'cat_regression':
            model = CatRegressor.load(os.path.join(args.dir_path, dir_name, exp_name))
        elif args.model_type == 'cat_ranking':
            model = CatRanker.load(os.path.join(args.dir_path, dir_name, exp_name))
        else:
            raise NotImplementedError
        if args.use_op_split:
            data_prefix = os.path.join('split_tuning_dataset_op', dir_name, exp_name)
        else:
            data_prefix = os.path.join('split_tuning_dataset', dir_name, exp_name)
        test_df = read_pd(data_prefix + '.test.pq')
        with open(data_prefix + '.used_key.json', 'r') as in_f:
            used_key = json.load(in_f)
        test_df = test_df[used_key]
        group_indices = get_group_indices(test_df)
        if args.eval_correlation:
            test_features, test_labels = get_feature_label(test_df)
            valid_indices = (test_labels > 0).nonzero()[0]
            test_scores = model.predict(test_features)
            ndcg_top_2 = ndcg_score(np.expand_dims(test_labels, axis=0),
                                    np.expand_dims(test_scores, axis=0), k=2)
            ndcg_top_5 = ndcg_score(np.expand_dims(test_labels, axis=0),
                                    np.expand_dims(test_scores, axis=0), k=5)
            ndcg_top_8 = ndcg_score(np.expand_dims(test_labels, axis=0),
                                    np.expand_dims(test_scores, axis=0), k=8)
            ndcg_top_10 = ndcg_score(np.expand_dims(test_labels, axis=0),
                                     np.expand_dims(test_scores, axis=0), k=10)

            ndcg_group_avg_2 = group_ndcg_score(test_labels, test_scores,
                                                k=2,
                                                group_indices=group_indices)
            ndcg_group_avg_5 = group_ndcg_score(test_labels, test_scores,
                                                k=5,
                                                group_indices=group_indices)
            ndcg_group_avg_8 = group_ndcg_score(test_labels, test_scores,
                                                k=8,
                                                group_indices=group_indices)
            ndcg_group_avg_10 = group_ndcg_score(test_labels, test_scores,
                                                 k=10,
                                                 group_indices=group_indices)
            pearson_score, _ = pearsonr(test_scores, test_labels)
            spearman_score, _ = spearmanr(test_scores, test_labels)
            noninvalid_pearson_score, _ = pearsonr(test_scores[valid_indices],
                                                   test_labels[valid_indices])
            noninvalid_spearman_score, _ = spearmanr(test_scores[valid_indices],
                                                     test_labels[valid_indices])
            if not args.ranking_only:
                rmse = np.sqrt(np.square(test_labels - test_scores).mean())
                mae = np.abs(test_labels - test_scores).mean()
            ele_results = [f'{dir_name}/{exp_name}',
                                    spearman_score,
                                    pearson_score,
                                    noninvalid_spearman_score,
                                    noninvalid_pearson_score,
                                    ndcg_top_2,
                                    ndcg_top_5,
                                    ndcg_top_8,
                                    ndcg_top_10,
                                    ndcg_group_avg_2,
                                    ndcg_group_avg_5,
                                    ndcg_group_avg_8,
                                    ndcg_group_avg_10]
            if not args.ranking_only:
                ele_results.append(rmse)
                ele_results.append(mae)
            correlation_dat.append(ele_results)
        else:
            rank_test_all = np.load(data_prefix + '.rank_test.all.npz')
            rank_test_valid = np.load(data_prefix + '.rank_test.valid.npz')
            test_features, test_labels = get_feature_label(test_df)
            test_score = model.evaluate(test_features, test_labels, 'regression')
            test_ranking_score_all = model.evaluate(rank_test_all['rank_features'],
                                                    rank_test_all['rank_labels'],
                                                    'ranking')
            test_ranking_score_all = {k + '_all': v for k, v in test_ranking_score_all.items()}
            test_score.update(test_ranking_score_all)
            test_ranking_score_valid = model.evaluate(rank_test_valid['rank_features'],
                                                      rank_test_valid['rank_labels'],
                                                      'ranking')
            test_ranking_score_valid = {k + '_valid': v for k, v in
                                        test_ranking_score_valid.items()}
            test_score.update(test_ranking_score_valid)
            print(test_score)
            with open(os.path.join(os.path.join(args.dir_path, dir_name, exp_name),
                                   'test_scores.json'),
                      'w') as out_f:
                json.dump(test_score, out_f)
if args.eval_correlation:
    columns = ['name', 'spearman', 'pearson', 'spearman_v', 'pearson_v',
               'ndcg-2', 'ndcg-5', 'ndcg-8', 'ndcg-10',
               'ndcg-group-2', 'ndcg-group-5', 'ndcg-group-8', 'ndcg-group-10']
    if not args.ranking_only:
        columns += ['rmse', 'mae']
    out_df = pd.DataFrame(correlation_dat, columns=columns)
    out_df.to_csv(args.correlation_out_name + '.csv')
