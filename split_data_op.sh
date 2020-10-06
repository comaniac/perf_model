set -ex
OUT_DIR=split_tuning_dataset_op
mkdir -p ${OUT_DIR}

for fold in `cat tasks.txt`;
do
    mkdir -p ${OUT_DIR}/${fold}
    dataset=tuning_dataset/${fold}.csv
    split_train_name=${OUT_DIR}/$fold.train.pq
    split_test_name=${OUT_DIR}/$fold.test.pq
    python3 -m perf_model.thrpt_model_new  \
              --split_test_op_level \
              --dataset ${dataset} \
              --save_used_keys \
              --used_key_path ${OUT_DIR}/$fold.used_key.json \
              --split_train_name ${split_train_name} \
              --split_test_name ${split_test_name} \
              --split_test_ratio 0.1 \
              --seed 123
done;
