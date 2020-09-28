import pandas as pd
import argparse
import os
import json
import numpy as np
from perf_model.thrpt_model_new import NNRanker, read_pd, get_feature_label, CatRegressor,\
    CatRanker

parser = argparse.ArgumentParser(description='Read model results.')
parser.add_argument('--dir_path', type=str, default='model_results/nn_5.0_200')
parser.add_argument('--model_type', choices=['nn', 'cat_regression', 'cat_ranking'])
args = parser.parse_args()

for dir_name in sorted(os.listdir(args.dir_path)):
    for exp_name in sorted(os.listdir(os.path.join(args.dir_path, dir_name))):
        if args.model_type == 'nn':
            model = NNRanker.load(os.path.join(args.dir_path, dir_name, exp_name))
        elif args.model_type == 'cat_regression':
            model = CatRegressor.load(os.path.join(args.dir_path, dir_name, exp_name))
        elif args.model_type == 'cat_ranking':
            model = CatRanker.load(os.path.join(args.dir_path, dir_name, exp_name))
        else:
            raise NotImplementedError
        data_prefix = os.path.join('split_tuning_dataset', dir_name, exp_name)
        test_df = read_pd(data_prefix + '.test.pq')
        with open(data_prefix + '.used_key.json', 'r') as in_f:
            used_key = json.load(in_f)
        test_df = test_df[used_key]
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
        with open(os.path.join(os.path.join(args.dir_path, dir_name, exp_name), 'test_scores.json'), 'w') as out_f:
            json.dump(test_score, out_f)
