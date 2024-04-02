import json
import time
import pandas as pd
import logging
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator
import json
import os

class Evaluator:
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, model: BaseEstimator):
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

    def evaluate(self):
        """
        Evaluate a model
        """
        start_time = time.time()
        y_pred = self.model.predict(self.X_test)
        end_time = time.time()

        evaluation_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'inference_time': (end_time - start_time) / len(self.X_test)
        }

        logging.info(f"Results: {json.dumps(evaluation_metrics, indent=4)}")
        with open(os.path.join('models', 'last_trained_model_results.json'), 'w') as file:
            file.write(json.dumps(evaluation_metrics, indent=4))

        return evaluation_metrics