import pandas as pd
import pytest
import logging
import pickle
from joblib import load
import src.data
import numpy as np
from pandas.core.frame import DataFrame

logging.basicConfig(
    filename='./test/test_data.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

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

@pytest.fixture
def data():
    '''
    This fixture returns the data for the tests
    output:
            dataframe: pandas dataframe
    '''
    try:
        dataframe = pd.read_csv('data/clean_sample.csv')
        logging.info("import data: SUCCESS")
        return dataframe
    except FileNotFoundError as err:
        logging.error("import data: The file wasn't found")
        raise err


def test_data_shape(data):
    """ 
        If your data is in the correct shape
        input:
            data: pandas dataframe
    """
    assert data.shape[0] > 0, "The dataframe has no columns"
    assert data.shape[1] > 0, "The dataframe has no rows"

def test_columns_present(data):
    """ 
        If columns are present in dataframe
        input:
            data: pandas dataframe
    """
    for category in cat_features:
        assert category in data.columns, "The column {} is not in the dataframe".format(category)
        assert data[category].shape[0] > 0, "The column {} is empty".format(category)

def test_data_types(data):
    """ 
        If data types are correct
        input:
            data: pandas dataframe
    """
    for category in cat_features:
        assert data[category].dtype == 'object', "The column {} is not of type object".format(category)
        
def test_inference_below():
    """
    Check inference performance
    """
    model = load("model/model.pkl")
    encoder = load("model/encoder.enc")
    lb = load("model/lb.enc")


    array = np.array([[
                     19,
                     "Private",
                     "HS-grad",
                     "Never-married",
                     "Own-child",
                     "Husband",
                     "Black",
                     "Male",
                     40,
                     "United-States"
                     ]])
    df_temp = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = src.data.process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = src.common_functions.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
