import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        logger.debug('Data loaded and drop null value from %s', file_path)
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def add_lag(train_data, test_data,n_hours,m_days):
    try:
        for i in range(1, n_hours + 1):
            train_data[f'hour_lag_{i}'] = train_data['nat_demand'].shift(i)
        for i in range(1, m_days + 1):
            train_data[f'day_lag_{i}'] = train_data['nat_demand'].shift(i * 24)
        train_data['target'] = train_data['nat_demand']
        train_data.dropna(inplace=True)

        for i in range(1, n_hours + 1):
            test_data[f'hour_lag_{i}'] = test_data['nat_demand'].shift(i)
        for i in range(1, m_days + 1):
            test_data[f'day_lag_{i}'] = test_data['nat_demand'].shift(i * 24)
        test_data['target'] = test_data['nat_demand']
        test_data.dropna(inplace=True)

        return train_data, test_data
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='mycode\params.yaml')
        n_hours = params['feature_engineering']['n_hours']
        m_days = params['feature_engineering']['m_days']
        
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = add_lag(train_data, test_data,n_hours,m_days)

        save_data(train_df, os.path.join("./data", "processed", "train_final.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_final.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
