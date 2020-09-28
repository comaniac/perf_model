## Requirements

```
python3 -m pip install torch torchvision

# For M6 instance, you can install via
python3 -m pip install https://github.com/Lmy0217/PyTorch-aarch64/raw/master/torch-1.6.0a0%2B8d883f5-cp36-cp36m-linux_aarch64.whl 
```

## Download the Example Dataset and Pretrained Models

```bash
# Download the datasets
aws s3 cp --recursive s3://hyuz-shared-data/dataset_0726 tuning_dataset
aws s3 cp --recursive s3://xingjian-public/split_tuning_dataset_20200920 split_tuning_dataset

# The old model
aws s3 cp --recursive s3://hyuz-shared-data/trained_models trained_models

# Download the catboost models
aws s3 cp --recursive s3://xingjian-public/lorien/models/cat_regression_20200923/ model_results/cat_regression
aws s3 cp --recursive s3://xingjian-public/lorien/models/cat_ranking_20200923/ model_results/cat_ranking

# Download the NN models
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_0.0_40_20200926/ model_results/nn_0.0_40
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_0.0_80_20200926/ model_results/nn_0.0_80
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_1.0_40_20200926/ model_results/nn_1.0_40
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_1.0_80_20200926/ model_results/nn_1.0_80
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_1.0_40_hinge_20200926/ model_results/nn_1.0_40_hinge
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_1.0_80_hinge_20200926/ model_results/nn_1.0_80_hinge
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_1.0_40_approx_ndcg_20200926/ model_results/nn_1.0_40_approx_ndcg
aws s3 cp --recursive s3://xingjian-public/lorien/models/nn_1.0_80_approx_ndcg_20200926/ model_results/nn_1.0_80_approx_ndcg
```

The `split_tuning_dataset` is generated based on the tuning dataset. We can generate the dataset as follows:
```
bash split_data.sh
```

## Train Performance Models with Different Algorithms 
See more in [training_scripts](./training_scripts).

## Evaluate Performance Models with Different Algorithms 
See more in [evaluation_scripts](./training_scripts).

## Train Listwise Rank Model with Catboost

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
# Use the catboost regression model
python3 app/main.py --list-net ./model_results/cat_regression/gcv_t4_csv \
                    --model_type cat_regression \
                    --target "cuda -model=t4" --gcv MobileNetV2_1.0
python3 app/main.py --list-net ./model_results/nn_5.0_200/gcv_t4_csv \
                    --model_type nn \
                    --target "cuda -model=t4" --gcv MobileNetV2_1.0
python3 app/main.py --list-net ./trained_models/listwise_t4 \
                    --target "cuda -model=t4" --gcv MobileNetV2_1.0
python3 app/main.py --list-net ./skylake_models --target "llvm -mcpu=skylake-avx512" --gcv MobileNetV2_1.0
```

## Legacy

### Run Training

```bash
cd tests; sh train.sh
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

