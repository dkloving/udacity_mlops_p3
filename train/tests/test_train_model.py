import pytest
from pathlib import Path
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder

from train import train_model


@pytest.fixture
def dummy_data():
    X = np.zeros(shape=(10,4))
    y = np.ones(shape=(10,))
    return X, y


@pytest.fixture
def trained_model(dummy_data):
    X, y = dummy_data
    model = train_model.fit_model(X, y)
    return model


def test_fit_model(dummy_data, trained_model):
    """Checks that we can get a model of the expected type that is trained.
    """
    X, y = dummy_data
    assert type(trained_model) == AdaBoostClassifier
    
    preds = trained_model.predict(X)
    assert all(preds == y)

    
def test_save_model(tmpdir, trained_model):
    """Checks that we can save and load a model.
    """
    save_path = Path(tmpdir) / Path("tmp.joblib")
    train_model.save_model(save_path, [OneHotEncoder(), trained_model])
    model = joblib.load(save_path)
    assert type(model) == Pipeline

    
def test_eval_model(dummy_data, trained_model):
    """Check evaluation metrics are calculated correctly on dummy data
    """
    X, y = dummy_data
    precision, recall, fbeta = train_model.eval_model(X, y, trained_model)
    assert precision == 1.0 and recall == 1.0
