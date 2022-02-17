# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib
import logging

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model

# setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def prepare_data():
    logger.info("Reading data")
    data = pd.read_csv("data/census_clean.txt")
    
    logger.info("Preprocessing data")
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(
                                        train,
                                        categorical_features=cat_features,
                                        label="salary",
                                        training=True
                                    )
    X_test, y_test, encoder, lb = process_data(
                                        test,
                                        categorical_features=cat_features,
                                        label="salary",
                                        training=False,
                                        encoder=encoder,
                                        lb=lb
                                    )
    return X_train, y_train, X_test, y_test, encoder


def fit_model(X_train, y_train):
    logger.info("Training Model")
    model = train_model(X_train, y_train)
    return model


def save_model(save_dest, pipeline_objs=[]):
    # Train and save a model.
    logger.info("Saving pipeline to %s", save_dest)
    pipeline = make_pipeline(*pipeline_objs)
    joblib.dump(pipeline, save_dest)

if __name__ == '__main__':
    X_train, y_train, _, _, encoder = prepare_data()
    model = fit_model(X_train, y_train)
    
    save_dest = 'model/trained_pipeline.pkl'
    pipeline_objs = [encoder, model]
    save_model(save_dest, pipeline_objs)
