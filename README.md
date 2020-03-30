### Run Training

```bash
cd tests; sh train.sh
```

### Run Training with AutoGluon (not working yet)

```bash
cd tests; sh auto_train.sh
```


### Learning to Rank

```bash
python thrpt_model.py --dataset depthwise_conv2d_nchw.cuda-cuda-model-t4.csv --algo nn --gpu
```
