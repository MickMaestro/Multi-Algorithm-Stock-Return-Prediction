# Stock Return Prediction Using Machine Learning

---

## Overview

This project implements and compares multiple machine learning models for predicting stock returns using historical price data and technical indicators. The analysis focuses on German DAX stocks including SAP, Siemens, Allianz, BASF, and BMW.

---

## Features

- **Data Collection**: Automated stock and market data fetching using Yahoo Finance API
- **Feature Engineering**: Comprehensive technical indicator calculation including:
  - Moving averages (10, 50, 200 day periods)
  - Relative Strength Index (RSI)
  - Volatility measures
  - Volume indicators
  - Price differentials and momentum indicators
- **Model Comparison**: Four different machine learning approaches:
  - Linear Regression
  - Support Vector Machine (SVM) with multiple kernels
  - Random Forest with hyperparameter tuning
  - Neural Network with early stopping
- **Cross-Validation**: Time-series cross-validation for robust model evaluation
- **Visualisation**: Comprehensive plotting of predictions, feature importance, and model comparisons

---

## Requirements

### Dependencies

```
pandas
numpy
matplotlib
seaborn
tensorflow
yfinance
scikit-learn
keras
```

### Python Version
- Python 3.7 or higher
- TensorFlow 2.x compatible environment

---

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn tensorflow yfinance scikit-learn
```

---

## Usage

### Running the Analysis

Execute the main script:

```bash
python CF969_Assignment_2_2404237.py
```

### Configuration

The default configuration analyses the following German stocks:
- SAP.DE (SAP)
- SIE.DE (Siemens)
- ALV.DE (Allianz)
- BAS.DE (BASF)
- BMW.DE (BMW)

Against the DAX40 market index (^GDAXI).

To modify the stock selection, edit the `selected_stocks` list in the main section:

```python
selected_stocks = [
    'SAP.DE',    # Your stock choices here
    'SIE.DE',    
    # Add or remove as needed
]
```

---

## Methodology

### Data Processing

1. **Data Retrieval**: Downloads 5 years of daily stock data
2. **Data Cleaning**: Handles missing values using forward fill
3. **Feature Creation**: Generates 40+ technical indicators per stock
4. **Target Preparation**: Creates next-day return predictions

### Model Training

Each model undergoes:
- **Data Splitting**: 80% training, 20% testing with temporal ordering preserved
- **Feature Scaling**: StandardScaler normalisation for all features
- **Hyperparameter Tuning**: Grid search for optimal parameters where applicable
- **Cross-Validation**: 5-fold time-series cross-validation

### Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

---

## Output Files

The programme generates several visualisation files:

### Individual Stock Analysis
- `{STOCK}_feature_importance.png` - Feature importance comparison
- `{STOCK}_Linear_Regression_predictions.png` - Linear regression predictions
- `{STOCK}_Support_Vector_Machine_predictions.png` - SVM predictions
- `{STOCK}_Random_Forest_predictions.png` - Random forest predictions
- `{STOCK}_Neural_Network_predictions.png` - Neural network predictions
- `{STOCK}_nn_learning_curves.png` - Neural network training curves

### Cross-Model Comparison
- `model_comparison_mse.png` - MSE comparison across all models and stocks
- `model_comparison_mae.png` - MAE comparison across all models and stocks
- `model_comparison_r2.png` - R² comparison across all models and stocks

---

## Key Functions

### Data Processing Functions

- `fetch_and_prepare_data()` - Downloads and processes stock data
- `prepare_features_target()` - Separates features from target variables

### Model Training Functions

- `train_linear_regression()` - Linear regression with feature importance analysis
- `train_svm()` - SVM with adaptive kernel selection
- `train_random_forest()` - Random forest with hyperparameter optimisation
- `train_neural_network()` - Neural network with early stopping

### Analysis Functions

- `perform_cross_validation()` - Time-series cross-validation
- `compare_model_performance()` - Cross-model performance comparison
- `plot_predictions()` - Prediction visualisation

---

## Model Details

### Linear Regression
- Standard scikit-learn implementation
- Feature importance based on coefficient magnitudes
- Baseline model for comparison

### Support Vector Machine
- Adaptive kernel selection (Linear → RBF → Polynomial)
- Hyperparameter tuning based on prediction variance
- Robust to outliers

### Random Forest
- Hyperparameter tuning for number of estimators (100-500)
- Feature importance via impurity reduction
- Handles non-linear relationships

### Neural Network
- 3-layer architecture (64-32-16 neurons)
- 20% dropout for regularisation
- Adam optimiser with early stopping
- Prediction clipping to prevent extreme values

---

## Technical Indicators

The system calculates numerous technical indicators:

- **Trend Indicators**: Moving averages, price-to-MA ratios
- **Momentum Indicators**: RSI, lagged returns
- **Volatility Indicators**: Rolling standard deviation
- **Volume Indicators**: Volume moving averages and ratios
- **Price Action**: High-low differentials, daily ranges
- **Market Indicators**: Market return correlations

---

## Error Handling

The implementation includes comprehensive error handling:
- Missing data imputation
- Infinite value replacement
- Model training failure recovery
- Prediction clipping for extreme values
- Graceful degradation for failed stocks

---

## Performance Considerations

- **Memory Usage**: Processes stocks sequentially to manage memory
- **Computation Time**: Neural network cross-validation uses simplified architecture
- **Data Quality**: Automatic handling of missing and zero values
- **Numerical Stability**: Clipping and replacement of extreme values

---

## Limitations

- **Data Dependency**: Relies on Yahoo Finance API availability
- **Prediction Horizon**: Currently configured for next-day predictions only
- **Market Conditions**: Performance may vary significantly across different market regimes
- **Feature Engineering**: Manual feature selection may not capture all relevant patterns

---

## Future Enhancements

Potential improvements include:
- LSTM/GRU models for better time-series handling
- Extended prediction horizons (weekly/monthly)
- Additional technical indicators (MACD, Bollinger Bands)
- Ensemble methods combining multiple models
- Risk-adjusted performance metrics
- Real-time prediction capabilities

---

## Author
Ibukunoluwa Michael Adebanjo
---

## Licence

This project is created for academic purposes as part of coursework requirements.
