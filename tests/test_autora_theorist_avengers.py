import numpy as np
import pytest
from autora.theorist.autora_theorist_avengers import ParabolaRegression

def test_initialization():
    theorist = ParabolaRegression()
    assert theorist is not None

def test_fit():
    model = ParabolaRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model.fit(X, y)
    assert model.coefficients is not None
    assert len(model.coefficients) == 8  # 1 intercept + 2 linear + 2 log + 3 quadratic terms

def test_predict():
    model = ParabolaRegression()
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 3])
    model.fit(X_train, y_train)
    X_test = np.array([[7, 8], [9, 10]])
    predictions = model.predict(X_test)
    assert predictions.shape == (2,)

def test_construct_design_matrix():
    model = ParabolaRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    A = model._construct_design_matrix(X)
    assert A.shape == (3, 8)  # 3 samples, 1 intercept + 2 linear + 2 log + 3 quadratic terms

def test_print_eqn(capsys):
    model = ParabolaRegression()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model.fit(X, y)
    model.print_eqn()
    captured = capsys.readouterr()
    assert "Discovered equation: y =" in captured.out

if __name__ == "__main__":
    pytest.main()