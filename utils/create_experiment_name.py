from datetime import datetime
import mlflow

def create_experiment_name():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    existing_experiment_names = [exp.name for exp in mlflow.search_experiments()]
    suffix = 1
    while True:
        experiment_name = f"{current_time}_{suffix}"
        if experiment_name not in existing_experiment_names:
            return experiment_name
        suffix += 1