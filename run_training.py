import logging

from train import train_model


# setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if __name__ == "__main__":
    logger.info("Reading data")
    train_data, test_data = train_model.load_data()

    logger.info("Training Model")
    model, encoder, lb = train_model.fit_model(train_data)

    save_dest = "model/trained_pipeline.joblib"
    logger.info("Saving pipeline to %s", save_dest)
    train_model.save_model(save_dest, [encoder, model])

    logger.info("Evaluating Model")
    precision, recall, fbeta = train_model.eval_model(test_data, model, encoder, lb)
    logger.info(
        "Eval Metrics: precision %f | recall %f | fbeta: %f", precision, recall, fbeta,
    )

    slice_data_dest = "model/slice_output.html"
    logger.info("Evaluating sodel on slices and writing to %s", slice_data_dest)
    train_model.eval_slices(test_data, model, encoder, lb, slice_data_dest)
