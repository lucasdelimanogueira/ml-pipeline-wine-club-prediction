from typing import Tuple
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import Settings

class DataSplitter:
    def __init__(self, test_size: float = 0.1) -> None:
        self.test_size = test_size

    def split_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into training and test sets
        """

        logging.info("# Data Splitting")

        df['is_full_club'] = df['is_full_club'].map({'f': 0, 't': 1})

        X = df.drop(columns=['is_full_club'])
        y = df['is_full_club']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=Settings.get('RANDOM_STATE'), shuffle=True)
        
        logging.info(f"Training set size: {len(X_train)} samples")
        logging.info(f"Test set size: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test