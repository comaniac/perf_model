import pandas as pd
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Read model results.')
parser.add_argument('--dir_path', type=str, default='model_results/cat_regression')
parser.add_argument('--out_name', type=str, default='cat_regression')
parser.add_argument('--ranking_only', action='store_true')
args = parser.parse_args()
ranking_metrics = ['ndcg_all', 'ndcg_k3_all', 'mrr_all', 'ndcg_valid', 'ndcg_k3_valid', 'mrr_valid']
if args.ranking_only:
    columns = ['name'] + ranking_metrics
else:
    columns = ['name', 'rmse', 'mae'] + ranking_metrics
results = []
for dir_name in sorted(os.listdir(args.dir_path)):
    for exp_name in sorted(os.listdir(os.path.join(args.dir_path, dir_name))):
        with open(os.path.join(args.dir_path, dir_name, exp_name, 'test_scores.json'), 'r') as in_f:
            dat = json.load(in_f)
            if args.ranking_only:
                results.append((f'{dir_name}/{exp_name}',
                                dat['ndcg_all'], dat['ndcg_k3_all'], dat['mrr_all'],
                                dat['ndcg_valid'], dat['ndcg_k3_valid'], dat['mrr_valid']))
            else:
                results.append((f'{dir_name}/{exp_name}',
                                dat['rmse'], dat['mae'],
                                dat['ndcg_all'], dat['ndcg_k3_all'], dat['mrr_all'],
                                dat['ndcg_valid'], dat['ndcg_k3_valid'], dat['mrr_valid']))
df = pd.DataFrame(results, columns=columns)
out_name = f"{args.out_name}.csv"
df.to_csv(out_name)
