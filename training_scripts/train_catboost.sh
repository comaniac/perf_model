set -ex

model_type=$1
niter=$2
task=$3

data_prefix=../split_tuning_dataset/$task
MODEL_DIR=../model_results/${model_type}_${niter}
mkdir -p ${MODEL_DIR}
python3 -m perf_model.thrpt_model_new \
    --algo ${model_type} \
    --data_prefix ${data_prefix} \
    --niter $niter \
    --out_dir ${MODEL_DIR}/$task
cp ${TUNING_DATASET}/$task.meta ${MODEL_DIR}/$task/feature.meta
