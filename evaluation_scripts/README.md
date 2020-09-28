# Run End-to-end Performance Tuning with the Trained Performance Model

## G4
```bash
# Evaluate Catboost Regression on G4
bash evaluate_e2e.sh g4 cat_regression cat_regression

# Evaluate Catboost Ranking on G4
bash evaluate_e2e.sh g4 cat_ranking cat_ranking

# Evaluate LambdaRank
bash evaluate_e2e.sh g4 nn_lambda_rank_1.0_120 nn
```

## C5
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh c5 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh c5 cat_ranking cat_ranking
```

## C4
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh c4 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh c4 cat_ranking cat_ranking
```

## P3
```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh p3 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh p3 cat_ranking cat_ranking
```

## M6

```bash
# Evaluate Catboost Regression
bash evaluate_e2e.sh m6 cat_regression cat_regression

# Evaluate Catboost Ranking
bash evaluate_e2e.sh m6 cat_ranking cat_ranking
```
