from typing import Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
import logging
from config.config import Settings

class DataPreprocessor:
    def __init__(self) -> None:
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def convert_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert NaN values
        """
        logging.info("Converting NaN values")
        df['cc_paid'].fillna(df['order_total'], inplace=True)

        return df

    def label_encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Label encode categorical columns
        """
        logging.info("Encoding labels")
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def normalize(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Normalize numerical columns
        """
        logging.info("Normalizing data")
        df[columns] = self.scaler.fit_transform(df[columns])
        logging.info(f"Preprocessing normalization: Mean: {self.scaler.mean_} Std: {self.scaler.scale_}")
        return df
    
    def normalize_using_existing_scaler(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Normalize numerical columns using parameters from existing scaler
        """
        df[columns] = self.scaler.transform(df[columns])

        return df
    
    def resample(self, X, y, method=Settings.get('RESAMPLING_STRATEGY')):
        resampling_method = {
            'RandomOverSampler': RandomOverSampler,
            'ADASYN': ADASYN,
            'SMOTE': SMOTE,
            'NearMiss': NearMiss,
            'RandomUnderSampler': RandomUnderSampler
        }

        logging.info(f"Resampling strategy: {Settings.get('RESAMPLING_STRATEGY')}")

        if method in resampling_method.keys():
            X, y = resampling_method[method]().fit_resample(X, y)
        else:
            logging.info(f"Resampling was not applied (not found)")

        return X, y
    
    def _run(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run all preprocessing functions
        """
        
        logging.info("# Data Preprocessing")

        X_train = self.label_encode(X_train, columns=['active_subscriber', 'intro_tier_1', 'intro_completed_day_of_week', 'intro_completed_month', 'intro_completed_year', 'intro_delivery_day_of_week', 'intro_delivery_month', 'intro_delivery_year'])
        X_test = self.label_encode(X_test, columns=['active_subscriber', 'intro_tier_1', 'intro_completed_day_of_week', 'intro_completed_month', 'intro_completed_year', 'intro_delivery_day_of_week', 'intro_delivery_month', 'intro_delivery_year'])

        X_train = self.normalize(X_train, columns=['product_charged', 'bottle_charged', 'bottle_count', 'price_per_bottle', 'shipping_charged', 'order_additional_tax_total', 'order_total', 'cc_paid', 'total_cc_paid_less_taxes'])
        X_test = self.normalize_using_existing_scaler(X_test, columns=['product_charged', 'bottle_charged', 'bottle_count', 'price_per_bottle', 'shipping_charged', 'order_additional_tax_total', 'order_total', 'cc_paid', 'total_cc_paid_less_taxes'])

        X_train = self.convert_nan_values(X_train)
        X_test = self.convert_nan_values(X_test)

        logging.info("Resampling to balance classes")
        X_train, y_train = self.resample(X_train, y_train)
        

        return X_train, X_test, y_train, y_test