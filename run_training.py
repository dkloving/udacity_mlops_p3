import logging
import pandas as pd

from ml import model, data, slice


# setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if __name__ == "__main__":
    logger.info("Reading data")
    train_data, test_data = data.load_data()

    logger.info("Preprocessing data")
    X_train, y_train, lb = data.process_data(train_data, label=data.TARGET, training=True)

    logger.info("Training Model")
    clf = model.fit_classifier(X_train, y_train, data.CAT_FEATURES)

    save_dest = "model/trained_pipeline.joblib"
    logger.info("Saving pipeline to %s", save_dest)
    model.save_model(save_dest, clf)

    logger.info("Evaluating Model")
    X_test, y_test, _ = data.process_data(train_data, label=data.TARGET, training=False, lb=lb)
    precision, recall, fbeta = model.eval_model(X_test, y_test, clf)
    logger.info(
        "Eval Metrics: precision %f | recall %f | fbeta: %f", precision, recall, fbeta,
    )

    slice_data_dest = "model/slice_output.html"
    logger.info("Evaluating model on slices and writing to %s", slice_data_dest)
    slice.eval_slices(test_data, clf, lb, slice_data_dest, data.CAT_FEATURES)
