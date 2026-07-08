import json
import logging
import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import yaml
from  dvclive import Live 
# Ensure the logs dictory exits
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(parmas_file_path:str)->dict:
    """load the params.yaml file."""
    try:
        with open (parmas_file_path,'rb') as file:
            params = yaml.safe_load(file)
            logger.debug(f"params loaded :{parmas_file_path}")
            return params
    except FileNotFoundError as e:
        logger.error(f"File Doesn't found:{e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML error :{e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading params :{e}")
        raise

def load_model(file_path: str):
    """Load the tranied model from a file."""

    try:
        with open(file_path, "rb") as file:
            model = pkl.load(file)
        logger.debug(f"Model loaded from {file_path}")
        return model
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception:
        logger.error(f"Unexpected error occured while loading the model : {file_path}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load the data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"test dataset loaded: {file_path}")
        return df
    except pd.errors.ParserError:
        logger.error(f"Failed to parese the csv file : {file_path}")
        raise
    except Exception:
        logger.error(f"Unexpected error while loading test dataset : {file_path}")
        raise


def evaluation_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluation the model and return the evalution metrics."""
    try:
        y_pred = clf.predict(x_test)
        y_pred_prob = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        metrics_dict = {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc}
        logger.debug("MOdel evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error(f"Unexpected error while model evaluation:{e}")
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """SAve the evaluation metrics to a json file."""
    try:
        # Ensure the dict exits
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Metrics saved to {file_path}")

    except Exception as e:
        logger.error(f"Error occured while saving the metrics: {e}")
        raise


def main():
    try:
        params = load_params(parmas_file_path='params.yaml')

        clf = load_model("./models/model.pkl")
        test_data = load_data("./data/processed/test_tfidf.csv")

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        metrics = evaluation_model(clf, x_test, y_test)
        
        # Experince tracking using dvclive
        with Live(save_dvc_exp = True) as live:
            live.log_metric('accuracy',accuracy_score(y_test,y_test))
            live.log_metric('precision',precision_score(y_test,y_test))
            live.log_metric('recall',recall_score(y_test,y_test))

            live.log_params(params)

        save_metrics(metrics, "reports/metrics.json")

    except Exception as e:
        logger.error(f"failed to complete the model evalution processs: {e}")
        raise


if __name__ == "__main__":
    main()
