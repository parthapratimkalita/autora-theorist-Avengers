# autora-theorist-Avengers

## Search Algorithm

The search algorithm employed by our theorist involves fitting the provided dataset by using a quadratic function which
takes the shape of a parabola. Hence, we have defined the class containing the parabolic function as
**ParabolaRegression** class. The algorithm follows these steps:

1. **Data Preparation**: The input data ( X ) is converted to a 2D numpy array.
2. **Design Matrix Construction**: A design matrix ( A ) is constructed, which includes:
    - An intercept term (a column of ones).
    - Linear terms (the original features).
    - Log-transformed terms (log of the original features).
    - Quadratic terms (squares and cross-products of the features).
3. **Coefficient Calculation**: The coefficients are calculated using the least squares method by solving the normal
   equation
```math
A \beta = y
```
4. **Model Fitting**: The model is fitted to the data by storing the calculated coefficients.

The goodness of the fit of the equation to the data is determined using the following statistical measures which are
commonly used in regression:

- **R²:** The R² value measures the amount of variance present in the values of the dependent variable that can or cannot
  be properly explained by the independent variable. An R² value of 0 means the independent variables in question cannot
  explain the variances present in the dependent variable, whereas an R² value of 1 means that the independent variable
  can fully account for the variances in the dependent variable.

- **Mean Squared Error (MSE):** The MSE value denotes the mean of the squared error difference between the predicted
  output values or labels and the original output values or labels. In the case of the parabolic function used by our
  theorist for fitting the data, the MSE is calculated between the actual output data points and the data points present
  on the parabola curve calculated by our theorist. This MSE value needs to be minimized by finding optimal coefficient
  values for each of the input feature terms present in the final parabola equation.

## Search Space

The search space for the theorist includes a variety of potential equations formed by different combinations of the
following terms:

- **Intercept Term**: A constant term.
- **Linear Terms**: The original features of the dataset.
- **Log-Transformed Terms**: The natural logarithm of the original features.
- **Quadratic Terms**: The squares and cross-products of the original features.

The search is constrained by the structure of the design matrix, which ensures that all relevant terms are included
without overfitting. The constraints include:

- **Feature Selection**: All features are considered in their original, log-transformed, and quadratic forms.
- **Equation Complexity**: The design matrix is constructed to include all necessary terms while avoiding unnecessary
  complexity.

These constraints help in efficiently exploring the search space and identifying the best-fitting quadratic equation.

**Example**

For an input matrix ( X ) with 2 features:
```math
X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
```

The design matrix ( A ) would be constructed as follows:
1. Intercept Term:
```math
\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
```
2. Linear Terms:
```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
```
3. Log-Transformed Terms:
```math
\begin{bmatrix} \log(1 + 1e-8) & \log(2 + 1e-8) \\ \log(3 + 1e-8) & \log(4 + 1e-8) \\ \log(5 + 1e-8) & \log(6 + 1e-8) \end{bmatrix}
```
4. Quadratic Terms:
```math
\begin{bmatrix} 1^2 & 1 \cdot 2 & 2^2 \\ 3^2 & 3 \cdot 4 & 4^2 \\ 5^2 & 5 \cdot 6 & 6^2 \end{bmatrix}
```

Combining all these terms, the final design matrix ( A ) would be:
```math
A = \begin{bmatrix} 1 & 1 & 2 & \log(1 + 1e-8) & \log(2 + 1e-8) & 1 & 2 & 4 \\ 1 & 3 & 4 & \log(3 + 1e-8) & \log(4 + 1e-8) & 9 & 12 & 16 \\ 1 & 5 & 6 & \log(5 + 1e-8) & \log(6 + 1e-8) & 25 & 30 & 36 \end{bmatrix}
```

This matrix ( A ) is then used to fit the quadratic model by solving the normal equation
```math
A \beta = y
```