# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model predicts whether a worker will earn greater than or less than $50,000 per year based on public census data.

The model file is a Scikit-Learn AdaBoostClassifier with a OneHotEncoder prepended.

## Intended Use

This serves as a toy example and should not be used to make policy or personal decisions.

## Training Data

Training data was provided by Udacity, the original souce is not documented, as is other metadata (such as the year it was collected).

## Evaluation Data

20% of the training dataset was witheld for use as an evaluation set.

## Metrics

Precision: 0.767661
Recall: 0.609726
fbeta: 0.679639

## Ethical Considerations

With the absence of information on the source, collection methodology, and dates for the input data this should be considered a toy model only. It would not be ethical to use this for decision-making outside of strictly technical issues.

## Caveats and Recommendations

This model is adequate for its intended use, but only its intended use.
