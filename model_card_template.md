# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- This model predicts the salary range of an individual
- The model was built using a random forest classifier
- Developed by Ahiwe Onyebuchi Valentine in July, 2021. 

## Intended Use
- This was used in practicing the steps involved in deploying an ML model in a CI/CD pipeline
- This model is suitable for a production environment

## Training Data
- Training Data source - https://archive.ics.uci.edu/ml/datasets/census+income, training data split

## Evaluation Data
- Evaluation Data source - https://archive.ics.uci.edu/ml/datasets/census+income, test data split

## Metrics
- Metrics from the model:
    - precision: 0.7711
    - recall: 0.5902
    - fbeta: 0.6686

## Ethical Considerations
- There were no ethical considerations

## Caveats and Recommendations
- The model could be optimised for fairness
