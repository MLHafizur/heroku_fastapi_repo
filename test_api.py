from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Test data 1
test_data1 = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# Test data 2
test_data2 = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec_managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5178,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"}


def test_get_root():
    '''
        Test that the root returns a message
    '''
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Greeting": "Welcome!"}


def test_post_predict_less_than_50k():
    '''
        Test that predicted value is less than 50k
    '''
    json_data = { k.replace('-', '_'): v for k, v in test_data1.items() }
    r = client.post("/predict", json=json_data)
    assert r.status_code == 200
    assert r.json() == {'prediction': 'Salary <= 50k'}


def test_post_predict_greater_than_50k():
    '''
        Test that predicted value is greater than 50k
    '''
    json_data = { k.replace('-', '_'): v for k, v in test_data2.items() }
    r = client.post("/predict", json=json_data)
    assert r.status_code == 200
    assert r.json() == {'prediction': 'Salary > 50k'}
