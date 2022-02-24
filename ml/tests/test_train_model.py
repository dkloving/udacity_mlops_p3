import pytest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer

from ml import model, data


@pytest.fixture
def dummy_data():
    arr = np.zeros(shape=(10, 14))
    arr[:, -1] = 1
    df = pd.DataFrame(arr)
    df.columns = data.CAT_FEATURES + [f"num_{i}" for i in range(13-len(data.CAT_FEATURES))] + ["salary"]
    return df


@pytest.fixture
def processed_data(dummy_data):
    X, y, lb = data.process_data(dummy_data, label="salary")
    return X, y, lb


@pytest.fixture
def fitted_model(processed_data):
    X, y, _ = processed_data
    clf = model.get_classifier(data.CAT_FEATURES)
    clf.fit(X, y)
    return clf


def test_process_data(processed_data):
    """ Checks that process data correctly splits data into X and y
    """
    X, y, lb = processed_data
    assert X.shape[1] == 13
    assert y.shape[0] == 10
    assert type(lb) == LabelBinarizer


def test_fit_model(fitted_model):
    """Checks that we can get and train a model of the expected type.
    """
    assert type(fitted_model) == Pipeline
    assert type(fitted_model.named_steps['model']) == AdaBoostClassifier
    assert type(fitted_model.named_steps['preprocessing']) == ColumnTransformer


def test_save_model(tmpdir):
    """Checks that we can save and load a model.
    """
    save_path = Path(tmpdir) / Path("tmp.joblib")
    clf = model.get_classifier(data.CAT_FEATURES)
    model.save_model(save_path, clf)
    pipeline = joblib.load(save_path)
    assert type(pipeline) == Pipeline


def test_eval_model(processed_data, fitted_model):
    """Check evaluation metrics are calculated correctly on dummy data.
    """
    X, y, _ = processed_data
    precision, recall, fbeta = model.eval_model(X, y, fitted_model)
    assert precision == 1.0 and recall == 1.0
