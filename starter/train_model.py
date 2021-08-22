# Script to train machine learning model.
"""
This Module contains multiple functions used to accomplish common tasks
in data science
This file can also be imported as a module and contains the following
functions:
    * import_data - returns dataframe for the csv found at pth

Author: Ahiwe Onyebuchi Valentine
Date: July 2021
"""

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import pandas as pd
import logging
import joblib

formatter = logging.Formatter('%(asctime)-15s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# main logger
logger = setup_logger('main_logger', 'main.txt')

# slice_output logger
slice_logger = setup_logger('slice_logger', 'slice_output.txt')

def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    try:
        dataframe = pd.DataFrame(pd.read_csv(pth))
        logger.info(
            "SUCCESS: Read file at %s with %s rows",
            pth,
            dataframe.shape[0])
    except FileNotFoundError as err:
        logger.error("ERROR: Failed to read file at %s", pth)
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        return dataframe
    except AssertionError as err:
        logger.error("The file doesn't appear to have rows and columns")
        raise err

def get_sliced_metrics(df, feature, model, enc, lb):
    """ Function for calculating performance on slices of the dataset.
        input:
            df: dataframe
            feature: feature to slice on
            model: model to use for inference
            enc: encoder for the feature
            lb: label encoder for the label
        output:
                dataframe: pandas dataframe
    """
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]
        X, y, _, _ = process_data(df_temp, categorical_features=cat_features, label="salary", training=False, encoder = enc, lb = lb)
        
        predicted_values = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, predicted_values)
        slice_logger.info(f"Feature: {feature}")
        slice_logger.info(f"Class: {cls}")
        slice_logger.info(f"{feature} precision: {precision:.4f}")
        slice_logger.info(f"{feature} recall: {recall:.4f}")
        slice_logger.info(f"{feature} fbeta: {fbeta:.4f}\n")


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

if __name__ == "__main__":
    try:
        # Add code to load in the data.
        data = import_data("../data/clean_sample.csv")

        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        train, test = train_test_split(data, test_size=0.20)

        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )

        # Proces the test data with the process_data function.
        X_test, y_test, encoder, lb = process_data(
            test, categorical_features=cat_features, label="salary", training=False,
            encoder = encoder, lb = lb
        )

        # Train and save a model.
        model = train_model(X_train, y_train)
        joblib.dump(model, "../model/model.pkl") 
        joblib.dump(encoder, "../model/encoder.enc")
        joblib.dump(lb, "../model/lb.enc")

        # Predictions on the test data.
        test_predicted_values = inference(model, X_test)

        # Calculate the metrics.
        precision, recall, fbeta = compute_model_metrics(y_test, test_predicted_values)

        # Print out the metrics.
        logger.info("Test Set Metrics:")
        logger.info(f"Test precision: {precision:.4f}")
        logger.info(f"Test recall: {recall:.4f}")
        logger.info(f"Test fbeta: {fbeta:.4f}\n")

        # # Load the model 
        model = joblib.load("../model/model.pkl") 
        enc = joblib.load("../model/encoder.enc")
        lb = joblib.load("../model/lb.enc")

        # Perform slice validation on native-country feature
        get_sliced_metrics(data, "native-country", model, enc, lb)

    except BaseException:
        logging.error("Model Training Failed")
        raise