import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ParabolaRegression(BaseEstimator):
    """
    Our Theorist
    """

    def __init__(self, alpha=1.0):
      self.alpha = alpha
      self.coef_ = None
      self.scaler = StandardScaler()

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
      y = np.asarray(y)

      y_log = np.log(y + 1e-8)

      # check if X is a numpy array
      if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")

      X_scaled = self.scaler.fit_transform(X)

      # Create the design matrix for quadratic regression
      # n_samples, n_features = X.shape

      X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

      # Start with the intercept term (column of ones)
      # A = self.construct_design_matrix(X)

      model = Ridge(alpha=self.alpha, fit_intercept=False)
      model.fit(X_scaled, y_log)
      self.coef_ = model.coef_.flatten()

      # Calculate the coefficients using the normal equation
      # self.coefficients = np.linalg.lstsq(A, y, rcond=None)[0]

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

      X_scaled = self.scaler.transform(X)

      X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

      # A = self.construct_design_matrix(X)

      y_log_pred = X_scaled @ self.coef_

      # Predict using the fitted coefficients
      # y_pred = A @ self.coefficients

      y_pred = np.exp(y_log_pred)

      return y_pred[:, np.newaxis]


    def construct_another_design_matrix(self, X):


    def construct_design_matrix(self, X):
      X = np.asarray(X)
      n_samples, n_features = X.shape
      A = np.ones((n_samples, 1))
      A = np.hstack([A, X])

      for i in range(n_features):
        for j in range(i, n_features):
          log_term = np.log(X[:, i] + 1e-8)
          A = np.hstack([A, log_term.reshape(-1, 1)])

      for i in range(n_features):
        for j in range(i, n_features):
          A = np.hstack([A, (X[:, i] * X[:, j]).reshape(-1, 1)])

      return A

    '''
        import numpy as npfrom
        scipy.optimize
        import curve_fit
        import matplotlib.pyplot as plt
    
        # Sample datax_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([1, 4, 9, 16, 25])
        t_data = np.array([0, 1, 2, 3, 4])  # Dynamic parameter values
    
        # Define the quadratic model with a dynamic constant
        def quadratic_model(x, a, b, c, d):
          return a * x ** 2 + b * x + c + d
    
        # Assume a simple linear function for d(t) as an example
        def dynamic_constant(t):
          return 2 * t  # Example dynamic function
    
        # Adjust y_data with dynamic constant
        y_adjusted = y_data - dynamic_constant(t_data)
        # Fit the quadratic model to the adjusted dataparams, covariance = curve_fit(quadratic_model, x_data, y_adjusted, p0=[1, 1, 1, 0])
        # Extract the coefficients
        a, b, c, d = params
        print(f"Coefficients:\na = {a}\nb = {b}\nc = {c}\nd = {d}")
        # Plot the original data and the fitted curveplt.scatter(x_data, y_data, label='Original Data')
        plt.plot(x_data, quadratic_model(x_data, *params), label='Fitted Quadratic Curve', color='red')
        plt.legend()
        plt.show()
    '''
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
