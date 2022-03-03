import requests


APP_URL = "https://gentle-thicket-08491.herokuapp.com/"


def test_api_get_root():
    r = requests.get(APP_URL)
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hai! OwO"}
    return r.status_code, r.json()


def test_get_negative_prediction():
    r = requests.post(
        f"{APP_URL}infer/",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50k"}
    return r.status_code, r.json()


def test_get_positive_prediction():
    r = requests.post(
        f"{APP_URL}infer/",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 217400,  # this is the only thing changed lol
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50k"}
    return r.status_code, r.json()


if __name__ == "__main__":
    print(test_api_get_root())
    print(test_get_negative_prediction())
    print(test_get_positive_prediction())
