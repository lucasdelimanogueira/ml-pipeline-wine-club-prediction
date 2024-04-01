# firstleaf-club-prediction
Machine Learning pipeline to predict if customers will subscribe based on their first order


# Find best model-hyperparameters
python3 main.py --train_size 0.1 --models random_forest svm xgboost --param_grids "{'random_forest': {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15]}}" "{'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}}" "{'xgboost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}}"

python3 main.py --models xgboost --param_grids "{'xgboost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}}"


