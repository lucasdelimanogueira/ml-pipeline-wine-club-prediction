import os
from dotenv import load_dotenv

load_dotenv(os.path.join('config', 'config.env'))

class Settings:
    settings = {
        'DATASET_PATH': os.getenv('DATASET_PATH'),
        'RANDOM_STATE': int(os.getenv('RANDOM_STATE')),
        'CROSS_VALIDATION_SIZE': int(os.getenv('CROSS_VALIDATION_SIZE')),
        'TUNING_SCORE_METRIC': os.getenv('TUNING_SCORE_METRIC'),
        'RESAMPLING_STRATEGY': os.getenv('RESAMPLING_STRATEGY')
    }

    @classmethod
    def get(cls, key):
        return cls.settings.get(key)