import argparse
import logging
import ast
from src.load_dataset import LoadDataset
from src.tuning import Tuner
from src.training import Trainer
from src.evaluating import Evaluator
from utils.create_experiment_name import create_experiment_name
import mlflow
import mlflow.sklearn
import os
from config.config import Settings
import shutil

def main():
    parser = argparse.ArgumentParser(description='Train and find the best model with hyperparameters')
    parser.add_argument('--models', nargs='+', choices=['random_forest', 'svm', 'xgboost'], help='Models to test', required=True)
    parser.add_argument('--param_grids', type=str, nargs='+', help='Hyperparameter grids for each model', required=True)
    parser.add_argument('--tuning_train_size', type=float, default=1, help='Train size ratio (between 0 and 1) to define the proportion of trainset will be used for the experiments')
    args = parser.parse_args()

    experiment_name = create_experiment_name()
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    log_file_path = os.path.join('mlruns', experiment_id, 'logging.log')
    logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%m/%d/%Y - %H:%M:%S",
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    
    logging.info("# Parsed arguments:")
    for arg in vars(args):
        logging.info("%s: %s", arg, getattr(args, arg))
    
    with mlflow.start_run():
        dataset_loader = LoadDataset(dataset_path=Settings.get('DATASET_PATH'))
        X_train, X_test, y_train, y_test = dataset_loader.prepare()
        param_grids = {model: ast.literal_eval(grid) for model, grid in zip(args.models, args.param_grids)}
        best_model_type, best_params = Tuner(X_train, y_train, models=args.models, param_grids=param_grids, tuning_train_size=args.tuning_train_size).run()        
        trained_model = Trainer(X_train, y_train, model_type=best_model_type, params=best_params).train()
        results = Evaluator(X_test=X_test, y_test=y_test, model=trained_model).evaluate()

        mlflow.log_param("environment_variables", {key: os.getenv(key) for key in os.environ})
        mlflow.log_param("normalization_parameters", {"mean": dataset_loader.preprocessor.scaler.mean_, "std": dataset_loader.preprocessor.scaler.scale_})
        mlflow.log_param("hyperparameters", best_params)
        mlflow.sklearn.log_model(trained_model, best_model_type)
        shutil.copy(log_file_path, os.path.join('models', 'last_model_logs.txt'))
        
        for metric_name, metric_value in results.items():
            mlflow.log_metric(metric_name, metric_value)
                              
if __name__ == "__main__":
    main()
