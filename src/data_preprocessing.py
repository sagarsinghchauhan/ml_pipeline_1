import logging
import os
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
# Ensure the 'logs' directory exits
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# File handler
log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizering , removing stopwords and puncation and streaming .
    """

    ps = PorterStemmer()
    # convert into the lowercase
    text = text.lower()
    # tokenize the text
    text = nltk.word_tokenize(text)
    # remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words("english") and word not in string.punctuation]
    # stem the words
    text = [ps.stem(word) for word in text]
    # join the tokens back into a single string
    return " ".join(text)


def preprocess_df(df, text_column="text", target_column="target"):
    """
    preprocess the dataframe by encoding the target column , removing duplicates and transforming the text column.
    """
    try:
        logger.debug("starting preprocessing the Dataframe")
        # encode the traget columns
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("target_column encoded")

        # removing the duplicates
        df = df.drop_duplicates(subset=[text_column], keep="first")
        logger.debug("duplicates removed")

        # apply text tranformating to the specified text column
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("transform the datafrme is done ")

        return df

    except KeyError as e:
        logger.error(f"Columns not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while preprocessing: {e}")
        raise


def main(text_column="text", target_column="target"):
    """
    Main function to load raw data , preprocess it and save the processsed data.
    """
    try:
        # fetch the dataset
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logger.debug("Train and Test data loaded")

        # transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)

        logger.debug(f"Processed data saved to : {data_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {e}")
        raise
    except Exception as e:
        logger.error(f" Faild to complete the data tranformation process: {e}")
        raise


if __name__ == "__main__":
    main()
