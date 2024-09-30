# Avengers - Theorist Challenge

## Overview
This project is a Python-based application that includes a quadratic model for predicting values based on input features. The project also includes a testing setup using GitHub Actions and supports multiple Python versions and operating systems.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
The project is organized into the following main parts:

### `challenge`
This folder contains the challenge-related scripts and data. It is used for testing and validating the model against specific challenges or datasets.

### `docs`
This folder contains the documentation for the project. It includes detailed descriptions of the modules, classes, and functions used in the project.

### `src`
This folder contains the source code of the project. The main components are:
- `autora/theorist/autora_theorist_avengers/__init__.py`: Contains the implementation of the quadratic model, including methods for fitting, predicting, and printing the model equation.

## Installation
To install the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. Install the required dependencies:
   ```sh
   pip install -e .
   ```

## Usage
To use the quadratic model, you can import the relevant classes and methods from the `src` folder. Here is an example:

```python
from autora.theorist.autora_theorist_avengers import ParabolaRegression

# Initialize the model
model = ParabolaRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict using the model
predictions = model.predict(X_test)

# Print the discovered equation
model.print_eqn()
```

## Testing
The project uses `pytest` for testing. To run the tests, use the following command:

```sh
pytest
```

The project also includes a GitHub Actions workflow (`.github/workflows/test-pytest.yml`) to automatically run tests on different Python versions and operating systems.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.