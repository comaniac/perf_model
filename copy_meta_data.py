import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Read model results.')
parser.add_argument('--dir_path', type=str, default='model_results/cat_ranking')
parser.add_argument('--tuning_data_path', type=str, default='tuning_dataset')
args = parser.parse_args()

for dir_name in os.listdir(args.dir_path):
    for model_dir in os.listdir(os.path.join(args.dir_path, dir_name)):
        meta_data_path = os.path.join(args.tuning_data_path, dir_name, model_dir + '.meta')
        shutil.copyfile(meta_data_path, os.path.join(args.dir_path, dir_name, model_dir,
                                                     'feature.meta'))
