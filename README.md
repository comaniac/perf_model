### Run Training

```bash
cd tests; sh train.sh
```

### Run Training with AutoGluon (not working yet)

```bash
cd tests; sh auto_train.sh
```


### Use AutoGluon Tabular

```bash
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda-cuda-model-t4.csv --algo auto
```

### Learning to Rank

```bash
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda-cuda-model-t4.csv --algo nn --gpu
```
