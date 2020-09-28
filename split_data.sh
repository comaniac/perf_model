set -ex
mkdir -p split_tuning_dataset

for fold in `ls tuning_dataset`;
do
    mkdir -p split_tuning_dataset/${fold}
    for fname in `ls tuning_dataset/$fold/*.csv`;
    do
        prefix_name=${fname:0:-4}
        python3 -m perf_model.thrpt_model_new  \
                --split_test \
                --save_used_keys \
                --used_key_path ${prefix_name}.used_key.json \
                --dataset $fname \
                --split_train_name ${prefix_name}.train.pq \
                --split_test_name ${prefix_name}.test.pq \
                --split_rank_test_prefix ${prefix_name}.rank_test \
                --seed 123
        mv ${prefix_name}.train.pq split_tuning_dataset/${fold}
        mv ${prefix_name}.test.pq split_tuning_dataset/${fold}
        mv ${prefix_name}.rank_test.* split_tuning_dataset/${fold}
        mv ${prefix_name}.used_key.json split_tuning_dataset/${fold}
    done;
done;
