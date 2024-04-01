from typing import Any
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

class DataPreprocessor:
    def __init__(self) -> None:
        self.label_encoders = {}
        self.scaler = MinMaxScaler()

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
    
    def _run(self, train, val, test):
        
        logging.info("# Data Preprocessing")

        logging.info("Encoding labels")
        train = self.label_encode(train, columns=['active_subscriber', 'intro_tier_1', 'intro_completed_day_of_week', 'intro_completed_month', 'intro_completed_year', 'intro_delivery_day_of_week', 'intro_delivery_month', 'intro_delivery_year'])
        val = self.label_encode(val, columns=['active_subscriber', 'intro_tier_1', 'intro_completed_day_of_week', 'intro_completed_month', 'intro_completed_year', 'intro_delivery_day_of_week', 'intro_delivery_month', 'intro_delivery_year'])
        test = self.label_encode(test, columns=['active_subscriber', 'intro_tier_1', 'intro_completed_day_of_week', 'intro_completed_month', 'intro_completed_year', 'intro_delivery_day_of_week', 'intro_delivery_month', 'intro_delivery_year'])

        logging.info("Normalizing data")
        train = self.normalize(train, columns=['product_charged', 'bottle_charged', 'bottle_count', 'price_per_bottle', 'shipping_charged', 'order_additional_tax_total', 'order_total', 'cc_paid', 'total_cc_paid_less_taxes'])
        val = self.normalize_using_existing_scaler(val, columns=['product_charged', 'bottle_charged', 'bottle_count', 'price_per_bottle', 'shipping_charged', 'order_additional_tax_total', 'order_total', 'cc_paid', 'total_cc_paid_less_taxes'])
        test = self.normalize_using_existing_scaler(test, columns=['product_charged', 'bottle_charged', 'bottle_count', 'price_per_bottle', 'shipping_charged', 'order_additional_tax_total', 'order_total', 'cc_paid', 'total_cc_paid_less_taxes'])

        return train, val, test