NUM_GPU=8
i=0

for num_hidden in 32 64 128 256
do
  for num_layers in 1 2 3 4
  do
    for threshold in 1 5 10
    do
      screen -d -m python ../thrpt_model.py --dataset ../depthwise_conv2d_nchw.cuda-cuda-model-t4.csv \
        --algo nn \
        --gpus ${i} \
        --threshold ${threshold} --num_hidden ${num_hidden} --num_layers ${num_layers}
      ((i=(i+1)%NUM_GPU))
    done
  done
done
