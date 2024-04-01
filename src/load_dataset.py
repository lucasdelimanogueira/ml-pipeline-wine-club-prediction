from src.feature_engineering import FeatureEngineering
from src.data_preprocessing import DataPreprocessor
from src.data_splitting import DataSplitter
import pandas as pd
import logging

class LoadDataset:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path

    def prepare(self) -> pd.DataFrame:
        """
        Load and prepare dataset
        """
        df = pd.read_csv(self.dataset_path)
        df = FeatureEngineering()._run(df)
        X_train, X_test, y_train, y_test = DataSplitter().split_data(df)
        X_train, X_test = DataPreprocessor()._run(X_train, X_test)
        
        return X_train, X_test, y_train, y_test


