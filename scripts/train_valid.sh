dataset=$1
out_dir_par=$2
output_model=$out_dir_par/`basename ${dataset%.*}`
python3 perf_model/valid_model.py $dataset $output_dir/valid_net
mv valid.log $output_dir/log/

