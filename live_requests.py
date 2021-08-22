import requests
import json
url = "https://fastapi-ml-model.herokuapp.com"

request_data1 = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

request_data2 = {
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
    "native_country": "United-States"
}

response = requests.post("{}/predict/".format(url), data=json.dumps(request_data1))
print("Response code: ", response.status_code)
print("Response from API: ",response.json())

response = requests.post("{}/predict/".format(url), data=json.dumps(request_data2))
print("Response code: ", response.status_code)
print("Response from API: ",response.json())
