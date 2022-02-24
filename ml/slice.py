import logging
import pandas as pd

from . import model, data


# setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def slice_data(df, feature):
    """Slices data by holding a given categorical feature fixed."""
    for u in df[feature].unique():
        df_temp = df[df[feature] == u]
        yield df_temp, u


def eval_slices(eval_data, clf, lb, save_dest, slice_features):
    df = pd.DataFrame(columns=["Feature", "Value", "Precision", "Recall", "fbeta"])
    for feat in slice_features:
        for sample, feat_val in slice_data(eval_data, feat):
            X_slice, y_slice, _ = data.process_data(
                sample, label=data.TARGET, training=False, lb=lb
            )
            precision, recall, fbeta = model.eval_model(X_slice, y_slice, clf)
            df.loc[len(df.index)] = [feat, feat_val, precision, recall, fbeta]
    df.to_html(
        save_dest,
        index=False,
        float_format=lambda x: str(x)[:4],
        border=0,
        justify="left",
        col_space=80,
    )
    df.to_csv(save_dest.replace(".html", ".csv"), index=False)
