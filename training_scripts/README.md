# Run Training Scripts

```
# Run CatBoost Regression
cat tasks.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost.sh cat_regression 5000 1 8
# Run CatBoost Regression + 70% Training Data
cat tasks.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost.sh cat_regression 5000 0.7 8
# Run CatBoost Regression + 50% Training Data
cat tasks.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost.sh cat_regression 5000 0.5 8
# Run CatBoost Regression + 30% Training Data
cat tasks.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost.sh cat_regression 5000 0.3 8


# Run CatBoost Ranking
cat tasks.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost.sh cat_ranking 5000 1 8
# Run CatBoost Ranking + 70% Training Data
cat tasks.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost.sh cat_ranking 5000 0.7 8
# Run CatBoost Ranking + 50% Training Data
cat tasks.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost.sh cat_ranking 5000 0.5 8
```


### Run Neural Network Performance Model Ablation

Use a p3.16x instance.
```
# Baseline
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh -1 1000 512 3 0.1 0 1 8
# Baseline + Gate
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh -1 1000 512 3 0.1 1 1 8
# Baseline + Gate + Balanced
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh 2 1000 512 3 0.1 1 1 8


# Baseline + Gate + Balanced 70%
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh 2 1000 512 3 0.1 1 0.7 8
# Baseline + Gate + Balanced 50%
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh 2 1000 512 3 0.1 1 0.5 8
# Baseline + Gate + Balanced 30%
cat tasks.txt | awk '{print NR,$0}' | parallel -j 12 bash train_nn_regression_model.sh 2 1000 512 3 0.1 1 0.3 8

```

Rasp4b

```bash
# Baseline
cat rasp_tasks.txt | awk '{print NR,$0}' | parallel -j 4 bash train_nn_regression_model.sh -1 1000 512 3 0.1 0 1 8
# Baseline + Gate
cat rasp_tasks.txt | awk '{print NR,$0}' | parallel -j 4 bash train_nn_regression_model.sh -1 1000 512 3 0.1 1 1 8

# CatBoost Regression
cat rasp_tasks.txt | awk '{print NR,$0}' | parallel -j 4 bash train_catboost.sh cat_regression 5000 1 8

# CatBoost Ranking
cat rasp_tasks.txt | awk '{print NR,$0}' | parallel -j 4 bash train_catboost.sh cat_ranking 5000 1 8
```

Op-level split
```bash
# Baseline
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_nn_regression_model_op.sh -1 1000 512 3 0.1 0 1 8
# Baseline + Gate
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_nn_regression_model_op.sh -1 1000 512 3 0.1 1 1 8

# CatBoost Regression
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_regression 5000 1 8

# CatBoost Ranking
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_ranking 5000 1 8
```
Op-level split different ratio
```bash
# Train with 0.3 ratio
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_nn_regression_model_op.sh -1 1000 512 3 0.1 1 0.3 8
# Train with 0.5 ratio
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_nn_regression_model_op.sh -1 1000 512 3 0.1 1 0.5 8
# Train with 0.7 ratio
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_nn_regression_model_op.sh -1 1000 512 3 0.1 1 0.7 8
```

Op-level split ratio script catboost
```bash
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_regression 5000 0.3 8
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_regression 5000 0.5 8
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_regression 5000 0.7 8

cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_ranking 5000 0.3 8
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_ranking 5000 0.5 8
cat tasks_op.txt | awk '{print NR,$0}' | parallel -j 8 bash train_catboost_op.sh cat_ranking 5000 0.7 8
```
