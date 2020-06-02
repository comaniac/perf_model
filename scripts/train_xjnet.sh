dataset=$1
metadata=$2
out_dir_par=$3
out_dir=$out_dir_par/`basename ${dataset%.*}`
mkdir -p $out_dir_par

# Train rank net
python3 perf_model/thrpt_model.py --dataset $dataset --algo nn --gpus 0 --niter 1500000 --out_dir $out_dir
mkdir $out_dir/log
mv $out_dir/*.csv $out_dir/log
mv $out_dir/*.log $out_dir/log

# Train valid net
python3 perf_model/valid_model.py $dataset $out_dir/valid_net
mv valid.log $out_dir/log/

# Copy feature metadata
cp $metadata $out_dir/feature.meta

