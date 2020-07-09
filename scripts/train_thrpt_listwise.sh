dataset=$1
out_dir_par=$2
out_dir=$out_dir_par/`basename ${dataset%.*}`
mkdir $out_dir_par
python3 perf_model/thrpt_model.py --dataset $dataset --algo cat --niter 3000 --rank_loss_function YetiRank --out_dir $out_dir
mkdir $out_dir/log
mv $out_dir/* $out_dir/log
mv $out_dir/log/*.cbm $out_dir
