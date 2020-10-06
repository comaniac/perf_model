set -ex

SAVE_DIR=evaluation_results
MODEL_BASE_DIR=model_results

mkdir -p ${SAVE_DIR}

#for model in cat_regression_split0.3 cat_regression_split0.5 cat_regression_split0.7 cat_regression_split1
for model in cat_regression_split1
do
  python3 evaluate.py --eval_correlation --dir_path ${MODEL_BASE_DIR}/${model} --model_type cat_regression --correlation_out_name ${SAVE_DIR}/${model}
done

python3 evaluate.py --eval_correlation --dir_path ${MODEL_BASE_DIR}/cat_ranking --model_type cat_ranking --correlation_out_name ${SAVE_DIR}/cat_ranking

for model in nn_regression_split0.3_-1_1000_512_3_0.1_0 \
             nn_regression_split0.3_-1_1000_512_3_0.1_1 \
             nn_regression_split0.5_-1_1000_512_3_0.1_0 \
             nn_regression_split0.5_-1_1000_512_3_0.1_1 \
             nn_regression_split0.7_-1_1000_512_3_0.1_0 \
             nn_regression_split0.7_-1_1000_512_3_0.1_1 \
             nn_regression_split1_-1_1000_512_3_0.1_0 \
             nn_regression_split1_-1_1000_512_3_0.1_1
do
  python3 evaluate.py --eval_correlation --dir_path ${MODEL_BASE_DIR}/${model} --model_type nn --correlation_out_name ${SAVE_DIR}/${model}
done
