import pytest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from train import train_model


@pytest.fixture
def dummy_data():
    arr = np.zeros(shape=(10, 9))
    arr[:, -1] = 1
    df = pd.DataFrame(arr)
    df.columns = train_model.CAT_FEATURES + ['salary']
    return df


@pytest.fixture
def trained_model(dummy_data):
    model, encoder, lb = train_model.fit_model(dummy_data)
    return model, encoder, lb


def test_fit_model(dummy_data, trained_model):
    """Checks that we can get a model of the expected type.
    """
    model, encoder, lb = trained_model
    assert type(model) == AdaBoostClassifier
    assert type(encoder) == OneHotEncoder
    assert type(lb) == LabelBinarizer


def test_save_model(tmpdir, trained_model):
    """Checks that we can save and load a model.
    """
    save_path = Path(tmpdir) / Path("tmp.joblib")
    model, encoder, lb = trained_model
    train_model.save_model(save_path, [encoder, model])
    pipeline = joblib.load(save_path)
    assert type(pipeline) == Pipeline


def test_eval_model(dummy_data, trained_model):
    """Check evaluation metrics are calculated correctly on dummy data.
    """
    precision, recall, fbeta = train_model.eval_model(dummy_data, *trained_model)
    assert precision == 1.0 and recall == 1.0
