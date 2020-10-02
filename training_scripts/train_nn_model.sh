set -ex

loss_type=$1
rank_lambda=$2
iter_mult=$3
task=$4

TUNING_DATASET=../tuning_dataset
data_prefix=../split_tuning_dataset/$task
MODEL_DIR=../model_results/nn_${loss_type}_${rank_lambda}_${iter_mult}
mkdir -p ${MODEL_DIR}
python3 -m perf_model.thrpt_model_new \
    --algo nn \
    --rank_loss_type ${loss_type} \
    --data_prefix ${data_prefix} \
    --rank_lambda ${rank_lambda} \
    --iter_mult ${iter_mult} \
    --out_dir ${MODEL_DIR}/$task
cp ${TUNING_DATASET}/$task.meta ${MODEL_DIR}/$task/feature.meta
