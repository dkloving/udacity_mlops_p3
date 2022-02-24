import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


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


TARGET = "salary"


def load_data(split_seed=42):
    data = pd.read_csv("data/census_clean.txt")
    train, test = train_test_split(data, test_size=0.20, random_state=split_seed)
    return train, test


def process_data(data: pd.DataFrame, label: str, training: bool = True, lb=None):
    """Process the data used in the machine learning pipeline.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label
    label : str
        Name of the label column in `data`.
    training : bool
        Indicator if training mode or inference/validation mode.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    y = data[label]
    X = data.drop([label], axis=1)

    if training is True:
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
    else:
        y = lb.transform(y.values).ravel()

    return X, y, lb
