dataset=$1
python3 thrpt_model.py --dataset $dataset --algo nn --gpus 0 --out_dir `basename ${dataset%.*}_nn_model`
