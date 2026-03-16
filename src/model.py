from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def svm_classifier(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
    """
    TODO: Return a scikit-learn SVC model with the specified parameters.
    """
    model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
    return model


def svm_regressor(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
    """
    TODO: Return a scikit-learn SVR model with the specified parameters.
    """
    model = SVR(kernel=kernel, C=C, degree=degree, gamma=gamma)
    return model

def evaluate_classifier(model, X_test, y_test):
    """
    TODO: Compute and return accuracy, precision, recall, and F1 score
    """
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    F1 = f1_score(y_test, prediction)
    
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": F1}


def evaluate_regressor(model, X_test, y_test):
    """
    TODO: Compute and return MAE, RMSE, and R2
    """
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    
    return {"Mean Absolute Error": mae, "Mean Squared Error": rmse, "R2": r2}