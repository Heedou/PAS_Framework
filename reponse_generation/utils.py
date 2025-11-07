import yaml
import pandas as pd
import os
from typing import Dict, Any

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    for key, value in config.get('environment', {}).items():
        os.environ[key] = value
    
    for path in config.get('paths', {}).get('create', []):
        os.makedirs(path, exist_ok=True)
    
    return config

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, encoding='utf-8-sig')

