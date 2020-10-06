import argparse
import os
import pandas as pd
import numpy as np
import glob
from perf_model.thrpt_model_new import get_group_df, get_group_indices

parser = argparse.ArgumentParser(description='downsample the op training dataset.')
parser.add_argument('--dir_path', type=str)
parser.add_argument('--out_path', type=str)
parser.add_argument('--ratio', type=float)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

info_l = []

np.random.seed(args.seed)
os.makedirs(args.out_path, exist_ok=True)
for folder in sorted(os.listdir(args.dir_path)):
    os.makedirs(os.path.join(args.out_path, folder), exist_ok=True)
    for name in sorted(os.listdir(os.path.join(args.dir_path, folder))):
        if name.endswith('.train.pq'):
            # Try to split
            train_df = pd.read_parquet(os.path.join(args.dir_path, folder, name))
            test_df = pd.read_parquet(os.path.join(args.dir_path, folder,
                                                   name[:-len('.train.pq')] + '.test.pq'))
            train_group_dfs = get_group_df(train_df)
            test_group_dfs = get_group_df(test_df)
            num_sampled_group = int(np.ceil(args.ratio * len(train_group_dfs)))
            perm = np.random.permutation(len(train_group_dfs))
            subsampled_train_df = pd.concat([train_group_dfs[perm[i]]
                                             for i in range(num_sampled_group)])
            print(folder, name, len(train_group_dfs), len(test_group_dfs))
            info_l.append((os.path.join(folder),
                           len(train_df), len(test_df),
                           len(train_group_dfs),
                           len(test_group_dfs)))
            subsampled_train_df.to_csv(os.path.join(args.out_path, folder, name))


stat_df = pd.DataFrame(info_l, columns=['name', 'train_num', 'test_num',
                                        'train_num_group', 'test_num_group'])
