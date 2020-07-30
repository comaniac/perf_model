### Train Listwise Rank Model with Catboost

#### Dataset

Starting from a set of AutoTVM tuning logs in JSON format (put them under `gcv_x86_c5_json` for example),
we need to run the following processes.

1. Featurize tuning logs and group by task names. The output would be `feature/gcv_x86_c5/`,
which includes a number of JSON files and one file is a dataset for a task.

```bash
python3 perf_model/data_proc.py featurize gcv_x86_c5
```

2. Standardize features and convert the JSON file datasets to CSV format.
The output would be `csv/`.

```bash
python3 perf_model/data_proc.py json2csv --std feature/gcv_x86_c5 -o gcv_skylake_csv
```

In `csv/`, each dataset contains two files:
- `task_name.csv`: The dataset file in CSV format for training.
- `tas_name.meta`: The dataset metadata, which includes the type of each feature (numeric or category) with its average and standard deviation. This is mainly used for inference.

#### Training

Based on the dataset we have processed, we can use the script to train a model for each dataset.
For example, the following command trains a model for task `conv2d_NCHWc.x86` and saves the results to `skylake/conv2d_NCHWc.x86`.

```bash
sh scripts/train_thrpt_listwise.sh gcv_skylake_csv/conv2d_NCHWc.x86.csv skylake
```

In `conv2d_NCHWc.x86`, we have:
- list_rank_net.cbm: The trained ranking model.
- log: Training logs and sampled dataset.

### Inference in AutoTVM

Before auto-tuning, we need to put the trained models in a specific way. For example:

```
skylake_models
|- conv2d_NCHWc.x86
  |- feature.meta
  |- list_rank_net.cbm
|- depthwise_conv2d_NCHWc.x86
  |- feature.meta
  |- list_rank_net.cbm
|- dense_pack.x86
  |- feature.meta
  |- list_rank_net.cbm
|- dense_nopack.x86
  |- feature.meta
  |- list_rank_net.cbm
```

As can be seen, each tuning task must have a corresponding folder name with trained model and feature metadata.
Now we can finally auto-tune a DNN model:

```bash
python3 app/main.py --list-net ./skylake_models --target "llvm -mcpu=skylake-avx512" --gcv MobileNetV2_1.0
```

## Legacy

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
aws s3 cp s3://hyuz-shared-data/gcv_t4_csv/depthwise_conv2d_nchw.cuda.csv .
```

### Use AutoGluon Tabular

```bash
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda.csv --algo auto --out_dir thrpt_autogluon
```

### Learning to Rank (Pairwise) + NN

```bash
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda.csv --algo nn --gpus 0 --out_dir thrpt_nn_model
```

### Learning to Rank (Listwise) with Catboost

```bash
pip install -U catboost --user
pip install -U scikit-learn>=0.23.1 --user
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda.csv \
                      --algo cat --rank_loss_function YetiRank \
                      --niter 3000 \
                      --out_dir thrpt_cat_model
```

Also, in [tune_params](tune_params), we have included a script for parameter tuning.

