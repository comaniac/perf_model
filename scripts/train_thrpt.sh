dataset=$1
out_dir_par=$2
out_dir=$out_dir_par/`basename ${dataset%.*}`
mkdir $out_dir_par
python3 perf_model/thrpt_model.py --dataset $dataset --algo nn --gpus 0 --niter 1500000 --out_dir $out_dir
mkdir $out_dir/log
mv $out_dir/*.csv $out_dir/log
mv $out_dir/*.log $out_dir/log

