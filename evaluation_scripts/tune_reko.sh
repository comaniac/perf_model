#reko/face_attribute/attributes
python3 ../app/tune_reko.py --list-net ../model_results/nn_regression_split1_-1_1000_512_3_0.1_1/gcv_v100_csv \
                            --n-parallel 8 \
                            --measure-top-n 8 \
                            --seed 123 \
                            --target "cuda -model=v100" \
