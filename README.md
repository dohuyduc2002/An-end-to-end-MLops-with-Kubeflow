# UnderWritingModel Project

## Overview
The UnderWritingModel project is designed to facilitate the development and evaluation of underwriting models using advanced machine learning techniques. The project leverages OptBinning for effective binning of categorical and numerical features, XGBoost as the primary training model, and SHAP for model interpretation. Additionally, MLflow is utilized for logging and tracking model performance.

## Project Structure
```
under_writing
├── src
│   ├── UnderWritingModel.py       # Contains the UnderWritingModel class
│   └── utils.py                    # Utility functions for data preprocessing
├── notebooks
│   └── under_writing_model_exploration.ipynb  # Jupyter notebook for model exploration
├── pyproject.toml                  # Project configuration and dependencies
├── README.md                       # Documentation for the project
└── .gitignore                      # Files and directories to ignore in Git
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can use the following commands:

```bash
git clone <repository-url>
cd under_writing
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Use the utility functions in `utils.py` for data preprocessing, including handling missing values and outlier detection.
2. **Model Training**: Instantiate the `UnderWritingModel` class from `UnderWritingModel.py`, fit the model on your training data, and make predictions.
3. **Model Evaluation**: Evaluate the model's performance using metrics logged with MLflow.
4. **Model Interpretation**: Use SHAP to explain the model's predictions and gain insights into feature importance.

## Example
Here is a brief example of how to use the `UnderWritingModel`:

```python
from src.UnderWritingModel import UnderWritingModel

# Initialize the model
model = UnderWritingModel()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Log metrics
model.log_metrics()

# Explain predictions
model.explain_predictions(X_test)
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.