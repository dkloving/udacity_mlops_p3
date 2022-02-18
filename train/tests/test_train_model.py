from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder

from train import train_model


def test_save_model(tmpdir):
    save_path = Path(tmpdir) / Path('tmp.joblib')
    train_model.save_model(save_path, [OneHotEncoder(), AdaBoostClassifier()])
    model = joblib.load(save_path)
    assert type(model) == Pipeline
