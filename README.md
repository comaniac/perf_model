### Run Training

```bash
cd tests; sh train.sh
```

### Run Training with AutoGluon (not working yet)

```bash
cd tests; sh auto_train.sh
```

### Download the Example Dataset

```bash
aws s3 cp s3://hyuz-shared-data/gcv_cuda_csv/depthwise_conv2d_nchw.cuda-cuda-model-t4.csv .
```

### Use AutoGluon Tabular

```bash
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda-cuda-model-t4.csv --algo auto --out_dir thrpt_autogluon
```

### Learning to Rank

```bash
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda-cuda-model-t4.csv --algo nn --gpus 0 --out_dir thrpt_nn_model
```

Also, in [tune_params](tune_params), we have included a script for parameter tuning.
