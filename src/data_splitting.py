from typing import Tuple
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 0) -> None:
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training, validation, and test sets
        """

        logging.info("# Data Splitting")

        train_data, test_data = train_test_split(df, test_size=self.test_size, random_state=self.random_state, shuffle=True)
        train_data, val_data = train_test_split(train_data, test_size=self.val_size, random_state=self.random_state, shuffle=True)

        return train_data, val_data, test_data

    def split_data_features_target(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into features and target, then split into training, validation, and test sets
        """
        features = df.drop(columns=[target_column])
        target = df[target_column]

        train_features, val_features, train_target, val_target = train_test_split(features, target, test_size=self.val_size, random_state=self.random_state, shuffle=True)

        train_features, val_features, test_features, train_target, val_target, test_target = train_test_split(train_features, train_target, test_size=self.test_size, random_state=self.random_state, shuffle=True)

        return train_features, val_features, test_features, train_target, val_target, test_target
