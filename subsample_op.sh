set -ex

for ratio in 0.3 0.5 0.7
do
  python3 downsample_op_data.py --dir_path split_tuning_dataset_op --out_path split_tuning_dataset_op_${ratio} --ratio ${ratio}
done
