# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib

# Add the necessary imports for the starter code.
from .ml.data import process_data, slice_data
from .ml.model import train_model, compute_model_metrics, inference


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_data():
    data = pd.read_csv("data/census_clean.txt")
    train, test = train_test_split(data, test_size=0.20)
    return train, test


def fit_model(train_data):
    X, y, encoder, lb = process_data(
        train_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    return model, encoder, lb


def eval_model(eval_data, model, encoder, lb):
    X, y, encoder, lb = process_data(
        eval_data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    return precision, recall, fbeta


def save_model(save_dest, pipeline_objs=[]):
    pipeline = make_pipeline(*pipeline_objs)
    joblib.dump(pipeline, save_dest)

    
def eval_slices(eval_data, model, encoder, lb, save_dest):
    df = pd.DataFrame(columns=["Feature", "Value", "Precision", "Recall", "fbeta"])
    for cat in CAT_FEATURES:
        for data, value in slice_data(eval_data, cat):
            precision, recall, fbeta = eval_model(data, model, encoder, lb)
            df.loc[len(df.index)] = [cat, value, precision, recall, fbeta]
    df.to_html(save_dest, index=False, float_format=lambda x: str(x)[:4], border=0, justify='left', col_space=80)
    df.to_csv(save_dest.replace('.html', '.csv'), index=False)
