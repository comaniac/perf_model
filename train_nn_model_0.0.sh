set -ex

rank_lambda=0.0
iter_mult=20

MODEL_DIR=model_results/nn_${rank_lambda}_${iter_mult}
mkdir -p ${MODEL_DIR}

TASKS=(
#"gcv_graviton2_csv/conv2d_NCHWc.x86"
#"gcv_graviton2_csv/conv2d_nchw_spatial_pack.arm_cpu"
#"gcv_graviton2_csv/conv2d_nchw_winograd.arm_cpu"
#"gcv_graviton2_csv/dense_nopack.x86"
#"gcv_graviton2_csv/depthwise_conv2d_NCHWc.x86"
#"gcv_graviton2_csv/depthwise_conv2d_nchw.arm_cpu"
#
#"gcv_skylake_csv/conv2d_NCHWc.x86"
#"gcv_skylake_csv/dense_nopack.x86"
#"gcv_skylake_csv/dense_pack.x86"
#"gcv_skylake_csv/depthwise_conv2d_NCHWc.x86"

#"gcv_t4_csv/conv2d_cudnn.cuda"
"gcv_t4_csv/conv2d_nchw.cuda"
"gcv_t4_csv/conv2d_nchw_winograd.cuda"
"gcv_t4_csv/conv2d_transpose_nchw.cuda"
#"gcv_t4_csv/dense_cublas.cuda"
#"gcv_t4_csv/dense_large_batch.cuda"
#"gcv_t4_csv/dense_small_batch.cuda"
"gcv_t4_csv/dense_tensorcore.cuda"
"gcv_t4_csv/depthwise_conv2d_nchw.cuda"
)


for task in ${TASKS[@]};
do
  data_prefix=split_tuning_dataset/$task
  python3 -m perf_model.thrpt_model_new \
      --algo nn \
      --data_prefix ${data_prefix} \
      --rank_lambda ${rank_lambda} \
      --iter_mult ${iter_mult} \
      --out_dir ${MODEL_DIR}/$task
done;
