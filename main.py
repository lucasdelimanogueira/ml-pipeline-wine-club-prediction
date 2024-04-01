import argparse
import logging
import ast
from src.load_dataset import LoadDataset
from src.tuning import Tuner
from src.training import Trainer
import pandas as pd
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Train and find the best model with hyperparameters')
    parser.add_argument('--models', nargs='+', choices=['random_forest', 'svm', 'xgboost'], help='Models to test', required=True)
    parser.add_argument('--param_grids', type=str, nargs='+', help='Hyperparameter grids for each model', required=True)
    parser.add_argument('--tuning_train_size', type=float, default=1, help='Train size ratio (between 0 and 1) to define the proportion of trainset will be used for the experiments')
    args = parser.parse_args()
    
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = LoadDataset(dataset_path="data/available_intro_user_data.csv").prepare()
        param_grids = {model: ast.literal_eval(grid) for model, grid in zip(args.models, args.param_grids)}
        best_model_type, best_params = Tuner(X_train, y_train, models=args.models, param_grids=param_grids, tuning_train_size=args.tuning_train_size).run()        
        trained_model = Trainer(X_train[:1000], y_train[:1000], model_type=best_model_type, params=best_params).train()
                
        mlflow.log_param("hyperparameters", best_params)
        mlflow.sklearn.log_model(trained_model, best_model_type)
        mlflow.log_metric("f1", 0.5)
        
if __name__ == "__main__":
    main()
