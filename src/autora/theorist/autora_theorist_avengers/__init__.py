from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ParabolaRegression(BaseEstimator):
    """
    Example Theorist
    """

    def __init__(self):
      self.coefficients = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]):
      """
      Fit the quadratic model to the data.

      X: array-like, shape (n_samples, n_features)
          Training data.
      y: array-like, shape (n_samples,)
          Target values.
      """

      # Ensure X is a 2D numpy array
      X = np.asarray(X)

      # Create the design matrix for quadratic regression
      n_samples, n_features = X.shape

      # Start with the intercept term (column of ones)
      A = np.ones((n_samples, 1))

      # Add the linear terms
      A = np.hstack([A, X])

      # Add the quadratic terms
      for i in range(n_features):
        for j in range(i, n_features):
          A = np.hstack([A, (X[:, i] * X[:, j]).reshape(-1, 1)])

      # Calculate the coefficients using the normal equation
      self.coefficients = np.linalg.lstsq(A, y, rcond=None)[0]

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
      """
      Predict using the quadratic model.

      X: array-like, shape (n_samples, n_features)
          Samples to predict.

      Returns
      -------
      y_pred: array, shape (n_samples,)
          Predicted values.
      """

      # Ensure X is a 2D numpy array
      X = np.asarray(X)

      # Create the design matrix for prediction
      n_samples, n_features = X.shape

      # Start with the intercept term (column of ones)
      A = np.ones((n_samples, 1))

      # Add the linear terms
      A = np.hstack([A, X])

      # Add the quadratic terms
      for i in range(n_features):
        for j in range(i, n_features):
          A = np.hstack([A, (X[:, i] * X[:, j]).reshape(-1, 1)])

      # Predict using the fitted coefficients
      y_pred = A @ self.coefficients

      return y_pred

    '''
    def print_eqn(self):
      """
      Print the discovered quadratic equation in human-readable format.
      """
      if self.coefficients is None:
        print("The model is not fitted yet.")
        return

      a, b, c = self.coefficients
      equation = f"y = {a.item():.4f} + {b.item():.4f}x + {c.item():.4f}x^2"
      print("Discovered equation:", equation)
    '''

    def print_eqn(self):
      """
      Print the discovered quadratic equation in human-readable format.
      """
      if self.coefficients is None:
        print("The model is not fitted yet.")
        return

      terms = ['1']  # Intercept term
      idx = 1
      n_features = int((len(self.coefficients) - 1) ** 0.5 * 2) - 1

      # Linear terms
      for i in range(n_features):
        terms.append(f'{self.coefficients[idx].item():.4f}*x{i + 1}')
        idx += 1

      # Quadratic terms
      for i in range(n_features):
        for j in range(i, n_features):
          terms.append(f'{self.coefficients[idx].item():.4f}*x{i + 1}*x{j + 1}')
          idx += 1

      equation = " + ".join(terms)
      print("Discovered equation: y =", equation)
