NUM_GPU=8
i=0
DATA_PATH=../depthwise_conv2d_nchw.cuda-cuda-model-t4.csv

for num_hidden in 128 256 512
do
  for num_layers in 2 3
  do
    for dropout in 0.05 0.1
    do
      for threshold in 1 5
      do
        screen -d -m python ../thrpt_model.py --dataset ${DATA_PATH} \
          --algo nn \
          --gpus ${i} \
          --threshold ${threshold} --num_hidden ${num_hidden} \
          --dropout ${dropout} \
          --num_layers ${num_layers} \
          --out_dir thrpt_h${num_hidden}_d${dropout}_l${num_layers}_tr${threshold}
        ((i=(i+1)%NUM_GPU))
      done
    done
  done
done
