import pandas as pd
import os
import logging
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from src.model import Model
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List
from config.config import Settings

class Tuner:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, models: List[str], param_grids: Dict[str, Dict[str, list]], tuning_train_size):
        self.models = models
        self.models = self.initialize_models()
        self.param_grids = param_grids
        self.tuning_train_size = tuning_train_size
        self.X_train = X_train
        self.y_train = y_train
        self.prepare_data()

    def prepare_data(self):

        total_train_size = len(self.X_train)
        if self.tuning_train_size < 1.0 and self.tuning_train_size > 0:
            self.X_train, _, self.y_train, _ = train_test_split(self.X_train, self.y_train, train_size=self.tuning_train_size, random_state=Settings.get('RANDOM_STATE'), shuffle=False)

        logging.info(f"Using {len(self.X_train)} / {total_train_size} samples for hyperparameter tuning")

    def initialize_models(self):
        return {model_type: Model(model_type) for model_type in self.models}

    def gridsearch(
        self, 
        model_type: List[BaseEstimator], 
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame
    ) -> Dict[str, Any]:
        
        logging.info(f"Training {model_type} model {self.param_grids[model_type][model_type]}")

        grid_search = GridSearchCV(
            self.models[model_type].model,
            self.param_grids[model_type][model_type],
            cv=Settings.get('CROSS_VALIDATION_SIZE'),
            scoring=Settings.get('TUNING_SCORE_METRIC'),
            verbose=True
        )
        grid_search.fit(X_train, y_train)
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

    def run(self):

        best_model_type = None
        best_score = -1
        for model_type in tqdm(self.models, desc="Training Models"):
            result = self.gridsearch(model_type, self.X_train, self.y_train)
            if result['best_score'] > best_score:
                best_model_type = model_type
                best_score = result['best_score']
                best_params = result['best_params']

        logging.info(f"Best model: {best_model_type}")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score ({Settings.get('TUNING_SCORE_METRIC')}): {best_score}")

        return best_model_type, best_params