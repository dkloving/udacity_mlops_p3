from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib


def get_classifier(categorical_columns):
    """
    Creates a pipeline to encode categorical features and then pass data to AdaBoost classifier.

    Parameters
    ----------
    categorical_columns: list[str]
        Column names to be one-hot encoded

    Returns
    -------
    pipeline
        Untrained sklearn pipeline
    """
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "OneHot",
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                categorical_columns,
            )
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        [("preprocessing", preprocessing), ("model", AdaBoostClassifier())]
    )
    return pipeline


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(clf, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    clf : sklearn.Pipeline
        Trained machine learning model.
    X : pd.DataFrame
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = clf.predict(X)
    return preds


def fit_classifier(X, y, categorical_features):
    clf = get_classifier(categorical_features)
    clf.fit(X, y)
    return clf


def save_model(save_dest, model):
    joblib.dump(model, save_dest)


def eval_model(X, y, clf):
    preds = inference(clf, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    return precision, recall, fbeta
