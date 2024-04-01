from typing import Union, Tuple
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
import logging

class Model:
    def __init__(self, model_type: str, **kwargs) -> None:
        """
        Initialize Model handler with selected model type and hyperparameters.

        Parameters:
        - model_type (str): Type of model to use ('random_forest', 'svm', or 'xgboost').
        - **kwargs: Additional hyperparameters for the model.
        """
        logging.info(f"# Initializing model - {model_type}")

        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == 'svm':
            self.model = SVC(**kwargs)
        elif model_type == 'xgboost':
            self.model = XGBClassifier(**kwargs)
        else:
            raise ValueError("Invalid model type. Choose from 'random_forest', 'svm', or 'xgboost'.")

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the selected model.

        Parameters:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series): Target variable.
        """
        logging.info(f"Training model")
        self.model.fit(X, y)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.

        Parameters:
        - filepath (str): Path to save the model.
        """
        logging.info(f"Saving model at {filepath}")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.

        Parameters:
        - filepath (str): Path to load the model from.
        """
        logging.info(f"Loading model from {filepath}")
        self.model = joblib.load(filepath)

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Make predictions using the trained model.

        Parameters:
        - X (pd.DataFrame): Feature matrix.

        Returns:
        - Predictions (pd.Series): Predicted values.
        """
        if isinstance(self.model, BaseEstimator):
            return self.model.predict(X)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            raise NotImplementedError("Prediction method not implemented for this model.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Evaluate the model on test data.

        Parameters:
        - X_test (pd.DataFrame): Test feature matrix.
        - y_test (pd.Series): Test target variable.

        Returns:
        - Evaluation score (float): Model evaluation score.
        """
        return self.model.score(X_test, y_test)