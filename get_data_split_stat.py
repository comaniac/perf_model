import argparse
import os
import pandas as pd
import numpy as np
import glob
from perf_model.thrpt_model_new import get_group_df, get_group_indices


parser = argparse.ArgumentParser(description='Get the statistics of the data split.')
parser.add_argument('--dir_path', type=str)
parser.add_argument('--out_path', type=str)
args = parser.parse_args()

info_l = []
for folder in sorted(os.listdir(args.dir_path)):
    for name in sorted(os.listdir(os.path.join(args.dir_path, folder))):
        if name.endswith('.train.pq'):
            # Try to split
            train_df = pd.read_parquet(os.path.join(args.dir_path, folder, name))
            test_df = pd.read_parquet(os.path.join(args.dir_path, folder,
                                                   name[:-len('.train.pq')] + '.test.pq'))
            train_group_indices = get_group_indices(train_df)
            test_group_indices = get_group_indices(test_df)
            info_l.append((os.path.join(folder),
                           len(train_df), len(test_df),
                           len(train_group_indices),
                           len(test_group_indices)))
stat_df = pd.DataFrame(info_l, columns=['name', 'train_num', 'test_num',
                                        'train_num_group', 'test_num_group'])
stat_df.to_csv(args.out_path)
