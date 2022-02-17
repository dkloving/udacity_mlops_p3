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

# Add code to load in the data.
logger.info("Reading data")
data = pd.read_csv("data/census_clean.txt")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

logger.info("Training Model")
X_train, y_train, encoder, lb = process_data(
                                    train,
                                    categorical_features=cat_features,
                                    label="salary",
                                    training=True
                                )

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
                                    test,
                                    categorical_features=cat_features,
                                    label="salary",
                                    training=False,
                                    encoder=encoder,
                                    lb=lb
                                )

# Train and save a model.
save_dest = 'model/trained_pipeline.pkl'
logger.info("Saving pipeline to %s", save_dest)
model = train_model(X_train, y_train)
pipeline = make_pipeline(encoder, model)
joblib.dump(pipeline, save_dest)
