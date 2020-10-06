import argparse
import glob
import os
import pandas as pd


parser = argparse.ArgumentParser(description='Read model results.')
parser.add_argument('--dir_path', type=str)
parser.add_argument('--out_path', type=str)
args = parser.parse_args()

score_columns = ['spearman', 'ndcg-2', 'ndcg-5', 'ndcg-8',
                 'ndcg-group-2', 'ndcg-group-5', 'ndcg-group-8']

out = []
index_l = []
for file_path in sorted(glob.glob(os.path.join(args.dir_path, '*.csv'))):
    model_name = file_path[:-len('.csv')]
    df = pd.read_csv(file_path)
    file_results = []
    for col in score_columns:
        avg_score = df[col].mean()
        file_results.append(avg_score)
    out.append(file_results)
    index_l.append(model_name)
out_df = pd.DataFrame(out, index=index_l, columns=score_columns)
out_df.to_csv(args.out_path)
