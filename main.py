import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib


CLASS_TABLE = ["<=50k", ">50k"]


class DataSample(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
            }
        }


def create_app():
    app = FastAPI()
    model = joblib.load("model/trained_pipeline.joblib")

    @app.get("/")
    async def read_root():
        return {"greeting": "Hai! OwO"}

    @app.post("/infer/")
    async def infer(sample: DataSample):
        df = pd.DataFrame.from_dict([sample.dict()])
        pred = model.predict(df)[0]
        return {"prediction": CLASS_TABLE[pred]}

    return app


# get model via DVC (thanks to https://ankane.org/dvc-on-heroku)
if "DYNO" in os.environ:
    os.system("dvc config core.no_scm true")
    os.system("dvc pull")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = create_app()
