import pandas as pd
import pytest
import logging
import pickle
from joblib import load
import src.common_functions

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
        
def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """
    encoder = load("model/encoder.enc")
    lb = load("model/lb.enc")

    X_test, y_test, _, _ = src.common_functions.process_data(
        data,
        categorical_features=src.common_functions.get_cat_features(),
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder(data):
    """
    Check split have same number of rows for X and y
    """
    encoder_test = load("model/encoder.enc")
    lb_test = load("model/lb.enc")

    _, _, encoder, lb = src.common_functions.process_data(
        data,
        categorical_features=src.common_functions.get_cat_features(),
        label="salary", training=True)

    _, _, _, _ = src.common_functions.process_data(
        data,
        categorical_features=src.common_functions.get_cat_features(),
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference_above():
    """
    Check inference performance
    """
    model = load("model/model.pkl")
    encoder = load("model/encoder.enc")
    lb = load("model/lb.enc")

    array = np.array([[
                     32,
                     "Private",
                     "Some-college",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "Black",
                     "Male",
                     80,
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

    X, _, _, _ = src.common_functions.process_data(
                df_temp,
                categorical_features=src.common_functions.get_cat_features(),
                encoder=encoder, lb=lb, training=False)
    pred = src.common_functions.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"


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

    X, _, _, _ = src.common_functions.process_data(
                df_temp,
                categorical_features=src.common_functions.get_cat_features(),
                encoder=encoder, lb=lb, training=False)
    pred = src.common_functions.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"
