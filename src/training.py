import pandas as pd
import logging
from typing import Dict, Any
from src.model import Model

class Trainer:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model_type: str, params=Dict[str, Any]):
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.params = params
        self.model = Model(model_type=self.model_type, **params)

    def train(self):
        logging.info(f"Training best model:{self.model_type} {self.params}")
        self.model.model.fit(self.X_train, self.y_train)
        self.model.save_model('models/last_trained_model.joblib')

        return self.model.model