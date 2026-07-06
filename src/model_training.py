import logging
import os
import pickle as pkl

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Ensure the logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load the data from a csv file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"data loaded from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected error occured while loading the data: {e}")
        raise


def train_model(train_data: pd.DataFrame, n_estimators: int) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the training data."""
    try:
        x_train = train_data.drop(columns=["label"]).values
        y_train = train_data["label"].values

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=2)
        clf.fit(x_train, y_train)
        logger.debug("model training completed")
        return clf
    except Exception as e:
        logger.error(f"unexpected error while training the model: {e}")
        raise


def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """Save the trained model to a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pkl.dump(model, f)
        logger.debug(f"model saved to {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving the model: {e}")
        raise


def main():
    try:
        n_estimators = 50
        train_data = load_data("./data/processed/train_tfidf.csv")
        clf = train_model(train_data, n_estimators)
        save_model(clf, os.path.join("./models", "model.pkl"))
    except Exception as e:
        logger.error(f"Failed to complete the model training process: {e}")
        raise


if __name__ == "__main__":
    main()
