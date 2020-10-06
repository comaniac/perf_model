# Run End-to-end Performance Tuning with the Trained Performance Model

## G4
```bash
# Evaluate Catboost Regression on G4
bash evaluate_e2e.sh g4 cat_regression_split1 cat_regression
bash evaluate_e2e.sh g4 cat_regression_split0.7 cat_regression
bash evaluate_e2e.sh g4 cat_regression_split0.5 cat_regression

# Evaluate Catboost Ranking on G4
bash evaluate_e2e.sh g4 cat_ranking cat_ranking

# Evaluate neural network
networks=(
nn_regression_split1_-1_1000_512_3_0.1_0
nn_regression_split1_-1_1000_512_3_0.1_1
nn_regression_split0.7_-1_1000_512_3_0.1_0
nn_regression_split0.7_-1_1000_512_3_0.1_1
nn_regression_split0.5_-1_1000_512_3_0.1_0
nn_regression_split0.5_-1_1000_512_3_0.1_1 
nn_regression_split0.3_-1_1000_512_3_0.1_0
nn_regression_split0.3_-1_1000_512_3_0.1_1
)

for model in ${networks[@]} 
do
    bash evaluate_e2e.sh g4 ${model} nn
done;

# Evaluate NN + Gate

for seed in 123 1234 12345
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;

# 0.3
for seed in 123 1234 12345
do
    for model in nn_regression_split0.3_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;

# 0.5
for seed in 123 1234 12345
do
    for model in nn_regression_split0.5_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;

# 0.7
for seed in 123 1234 12345
do
    for model in nn_regression_split0.7_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh g4 ${model} nn $K $seed
        done;
    done;
done;
```



## C5
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh c5 cat_regression_split1 cat_regression
bash evaluate_e2e.sh c5 cat_regression_split0.7 cat_regression
bash evaluate_e2e.sh c5 cat_regression_split0.5 cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh c5 cat_ranking cat_ranking

for seed in 123 1234 12345
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c5 ${model} nn $K $seed
        done;
    done;
done;

# Split Ratio 0.3
for seed in 123 1234 12345
do
    for model in nn_regression_split0.3_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c5 ${model} nn $K $seed
        done;
    done;
done;

# Split Ratio 0.5
for seed in 123 1234 12345
do
    for model in nn_regression_split0.5_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c5 ${model} nn $K $seed
        done;
    done;
done;

# Split Ratio 0.7
for seed in 123 1234 12345
do
    for model in nn_regression_split0.7_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c5 ${model} nn $K $seed
        done;
    done;
done;
```

## C4
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh c4 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh c4 cat_ranking cat_ranking

bash evaluate_e2e.sh c4 nn_regression_split1_-1_1000_512_3_0.1_0 nn 8
bash evaluate_e2e.sh c4 nn_regression_split1_-1_1000_512_3_0.1_1 nn 8


# Evaluate NN
for seed in 123 1234
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c4 ${model} nn $K $seed
        done;
    done;
done;

# 0.3
for seed in 123 1234 12345
do
    for model in nn_regression_split0.3_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c4 ${model} nn $K $seed
        done;
    done;
done;

# 0.5
for seed in 123 1234 12345
do
    for model in nn_regression_split0.5_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c4 ${model} nn $K $seed
        done;
    done;
done;

# 0.7
for seed in 123 1234 12345
do
    for model in nn_regression_split0.7_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh c4 ${model} nn $K $seed
        done;
    done;
done;
```

## P3
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh p3 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh p3 cat_ranking cat_ranking

# NN
bash evaluate_e2e.sh p3 nn_regression_split1_-1_1000_512_3_0.1_0 nn 2
bash evaluate_e2e.sh p3 nn_regression_split1_-1_1000_512_3_0.1_1 nn 2

# Multiple seed
for seed in 123 1234 12345
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh p3 ${model} nn $K $seed
        done;
    done;
done;

# 0.3
for seed in 123 1234 12345
do
    for model in nn_regression_split0.3_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh p3 ${model} nn $K $seed
        done;
    done;
done;

# 0.5
for seed in 123 1234 12345
do
    for model in nn_regression_split0.5_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh p3 ${model} nn $K $seed
        done;
    done;
done;

# 0.7
for seed in 123 1234 12345
do
    for model in nn_regression_split0.7_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh p3 ${model} nn $K $seed
        done;
    done;
done;
```

## M6

```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh m6 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh m6 cat_ranking cat_ranking

# NN Multiple Seeds
for seed in 123 1234 12345
do
    for model in nn_regression_split1_-1_1000_512_3_0.1_1
    do
        for K in 2 8
        do
            bash evaluate_e2e.sh m6 ${model} nn $K $seed
        done;
    done;
done;

```
