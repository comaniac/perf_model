dataset=$1
python3 perf_model/thrpt_model.py --dataset $dataset --algo nn --gpus 0 --out_dir `basename ${dataset%.*}_nn_model`
