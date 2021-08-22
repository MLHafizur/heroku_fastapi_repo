from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    param_grid = {'bootstrap': [True, False],
               'max_depth': [3, 5, 7, 10, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [2, 4, 8, 16, 32],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [50, 100, 150, 200]}
    rf = RandomForestClassifier(random_state = 123)
    model = RandomizedSearchCV(estimator = rf, 
                                   param_distributions = param_grid, 
                                   n_iter = 50, 
                                   cv = 3, 
                                   verbose=1, 
                                   random_state=42)
    model.fit(X_train, y_train)
    return model


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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
