import logging
import os
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the logs dirctory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger("feature-engineering")
logger.setLevel("DEBUG")

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, "feature_engineering.logs")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_params(params_path:str)-> dict:
    """Load parameter from a YAML file"""
    try:
        with open(params_path , 'rb') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Params loaded from :{params_path}")
        return params
    except FileNotFoundError as e :
        logger.error(f"File not found : {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Yaml error :{e}")
    except Exception as e:
        logger.error("Unexpected while loading params from : {params_path} : {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the data from the csv file.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug(f"data loaded and nans filled from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.debug(f"Failed to parse the csv file: {e}")
        raise
    except Exception as e:
        logger.debug(f"unexpected error occured while loading the data {e}")
        raise


def apply_tfid(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """appluy tfidf to the data"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train = train_data["text"].values
        y_train = train_data["target"].values
        x_test = test_data["text"].values
        y_test = test_data["target"].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df["label"] = y_train
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df["label"] = y_test

        logger.debug("tfidf applied and data tranformed")
        return train_df, test_df
    except Exception as e:
        logger.error(f"unexpected error while applying tfid: {e}")
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """save the dataframe to a csv file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"data saved to {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while savinf the data: {e}")
        raise


def main():
    try:
        params = load_params("params.yaml")
        max_feature = params['feature_engineering']['max_features']

        # max_feature = 50
        test_data = load_data("./data/interim/test_preprocessed.csv")
        train_data = load_data("./data/interim/train_preprocessed.csv")

        train_df, test_df = apply_tfid(train_data, test_data, max_feature)
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.debug(f"Failed to complete the feature enginnering process: {e}")
        raise


if __name__ == "__main__":
    main()
