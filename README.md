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

