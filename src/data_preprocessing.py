from typing import Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

class DataPreprocessor:
    def __init__(self) -> None:
        self.label_encoders = {}
        self.scaler = MinMaxScaler()

    def convert_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert NaN values
        """
        
        df['cc_paid'].fillna(df['order_total'], inplace=True)

        return df

    def label_encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Label encode categorical columns
        """
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def normalize(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Normalize numerical columns
        """
        df[columns] = self.scaler.fit_transform(df[columns])
        return df
    
    def normalize_using_existing_scaler(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Normalize numerical columns using parameters from existing scaler
        """
        df[columns] = self.scaler.transform(df[columns])

        return df
    
    def _run(
        self, 
        train: pd.DataFrame, 
        test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all preprocessing functions
        """
        
        logging.info("# Data Preprocessing")

        logging.info("Encoding labels")
        train = self.label_encode(train, columns=['active_subscriber', 'intro_tier_1', 'intro_completed_day_of_week', 'intro_completed_month', 'intro_completed_year', 'intro_delivery_day_of_week', 'intro_delivery_month', 'intro_delivery_year'])
        test = self.label_encode(test, columns=['active_subscriber', 'intro_tier_1', 'intro_completed_day_of_week', 'intro_completed_month', 'intro_completed_year', 'intro_delivery_day_of_week', 'intro_delivery_month', 'intro_delivery_year'])

        logging.info("Normalizing data")
        train = self.normalize(train, columns=['product_charged', 'bottle_charged', 'bottle_count', 'price_per_bottle', 'shipping_charged', 'order_additional_tax_total', 'order_total', 'cc_paid', 'total_cc_paid_less_taxes'])
        test = self.normalize_using_existing_scaler(test, columns=['product_charged', 'bottle_charged', 'bottle_count', 'price_per_bottle', 'shipping_charged', 'order_additional_tax_total', 'order_total', 'cc_paid', 'total_cc_paid_less_taxes'])

        logging.info("Converting NaN values")
        train = self.convert_nan_values(train)
        test = self.convert_nan_values(test)
        
        return train, test