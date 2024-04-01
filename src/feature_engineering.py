import pandas as pd
import logging

class FeatureEngineering:
    def __init__(self):
        self.functions = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__") and not func.startswith("_")]

    def convert_date_intro_completed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert dates of intro completed
        """

        logging.info("Converting dates of intro completed")

        df['intro_completed_day_of_week'] = pd.to_datetime(df['intro_completed_at'], format='%m/%d/%Y').dt.day_of_week
        df['intro_completed_month'] = pd.to_datetime(df['intro_completed_at'], format='%m/%d/%Y').dt.month
        df['intro_completed_year'] = pd.to_datetime(df['intro_completed_at'], format='%m/%d/%Y').dt.year

        return df
    
    def convert_date_intro_delivery(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte dates of intro delivery
        """

        logging.info("Converting dates of intro delivery")

        df['intro_delivery_day_of_week'] = pd.to_datetime(df['intro_delivery_date'], format='%m/%d/%Y').dt.day_of_week
        df['intro_delivery_month'] = pd.to_datetime(df['intro_delivery_date'], format='%m/%d/%Y').dt.month
        df['intro_delivery_year'] = pd.to_datetime(df['intro_delivery_date'], format='%m/%d/%Y').dt.year

        return df

    def days_to_delivery(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Number of days for the product to be delivered
        """
        
        logging.info("Creating feature days_to_delivery")

        df['intro_completed_at'] = pd.to_datetime(df['intro_completed_at'], format='%m/%d/%Y')
        df['intro_delivery_date'] = pd.to_datetime(df['intro_delivery_date'], format='%m/%d/%Y')
        df['days_to_delivery'] = (df['intro_delivery_date'] - df['intro_completed_at']).dt.days

        return df
    
    def _remove_unnecessary_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnnecessary fields that will not provide relevant information for the model
        """

        logging.info("Removing unnecessary fields")

        columns_to_remove = ['user_id', 'order_id', 'intro_completed_at', 'intro_delivery_date']
        df = df.drop(columns=columns_to_remove)

        return df
    
    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all the feature engineering functions
        """

        logging.info("# Feature Engineering")

        for function_name in self.functions:
            function = getattr(self, function_name)
            df = function(df)

        df = self._remove_unnecessary_fields(df)
            
        return df
