import numpy as np
from sklearn.base import BaseEstimator


class ParabolaRegression(BaseEstimator):
    """
    Our Theorist
    """

    def __init__(self):
      self.coefficients = None

    def fit(self, X: np.ndarray, y: np.ndarray):
      """
      Fit the quadratic model to the data.

      X: array-like, shape (n_samples, n_features)
          Training data.
      y: array-like, shape (n_samples,)
          Target values.
      """

      # Ensure X is a 2D numpy array
      X = np.asarray(X)

      # Check if X is a numpy array
      if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")

      # Construct the design matrix
      A = self._construct_design_matrix(X)

      # Calculate the coefficients using the normal equation
      self.coefficients = np.linalg.lstsq(A, y, rcond=None)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
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

      # Construct the design matrix
      A = self._construct_design_matrix(X)

      # Predict using the fitted coefficients
      return A @ self.coefficients

    def _construct_design_matrix(self, X):
      # Extract the number of samples and features
      n_samples, n_features = X.shape

      # Construct the design matrix
      A = np.ones((n_samples, 1))

      # Concatenate the original features, log-transformed features, and quadratic terms
      return np.hstack([A, X, self._log_transform(X), self._quadratic_transform(X)])

    def _log_transform(self, X: np.ndarray) -> np.ndarray:
      return np.log(X + 1e-8)

    def _quadratic_transform(self, X: np.ndarray) -> np.ndarray:
      n_samples, n_features = X.shape
      quadratic_terms = []

      for i in range(n_features):
        for j in range(i, n_features):
          quadratic_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))

      return np.hstack(quadratic_terms) if quadratic_terms else np.empty((n_samples, 0))

    def print_eqn(self):
      """
      Print the discovered quadratic equation in human-readable format.
      """
      if self.coefficients is None:
        print("The model is not fitted yet.")
        return

      # Flatten the coefficients array to a 1D array
      coef_flat = self.coefficients.flatten()

      # Calculate the number of features
      n_features = (len(self.coefficients) - 1) // 3

      # Initialize the equation list and index
      equation = []
      idx = 0

      # Add the intercept term
      equation.append(f'{coef_flat[idx]:.3f}')
      idx += 1

      # Add the linear terms
      for i in range(n_features):
        equation.append(f'{coef_flat[idx]:.3f} * x{i + 1}')
        idx += 1

      # Add the log-transformed terms
      for i in range(n_features):
        equation.append(f'{coef_flat[idx]:.3f} * log(x{i + 1} + 1e-8)')
        idx += 1

      # Add the quadratic terms
      for i in range(n_features):
        for j in range(i, n_features):
          equation.append(f'{coef_flat[idx]:.3f} * x{i + 1} * x{j + 1}')
          idx += 1

      # Construct and print the equation
      equation = " + ".join(equation)
      print("Discovered equation: y =", equation)
