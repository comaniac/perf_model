set -ex

model_type=$1
niter=$2
split_postfix=$3
num_gpus=$4
cuda_device_task=($5)
cuda_device=${cuda_device_task[0]}
task=${cuda_device_task[1]}


export CUDA_VISIBLE_DEVICES=$((${cuda_device} % ${num_gpus}))

TUNING_DATASET=../tuning_dataset
data_prefix=../split_tuning_dataset_op/$task
MODEL_DIR=../model_results/${model_type}_op_${niter}_split${split_postfix}
mkdir -p ${MODEL_DIR}
python3 -m perf_model.thrpt_model_new \
    --algo ${model_type} \
    --data_prefix ${data_prefix} \
    --niter $niter \
    --split_postfix ${split_postfix} \
    --out_dir ${MODEL_DIR}/$task
cp ${TUNING_DATASET}/$task.meta ${MODEL_DIR}/$task/feature.meta
