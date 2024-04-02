import os
from dotenv import load_dotenv

load_dotenv(os.path.join('config', 'config.env'))

class Settings:
    settings = {
        'DATASET_PATH': os.getenv('DATASET_PATH'),
        'RANDOM_STATE': int(os.getenv('RANDOM_STATE'))
    }

    @classmethod
    def get(cls, key):
        return cls.settings.get(key)