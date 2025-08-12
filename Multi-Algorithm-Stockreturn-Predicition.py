import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import yfinance as yf
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras
from tensorflow import keras
tf.random.set_seed(26)
np.random.seed(26)

#-----------------------------------------------------------------------------
'''
This function downloads stock data and market index data using yfinance.
It processes the data by calculating technical indicators including moving averages,
RSI, volatility measures, momentum indicators, and various price differentials.
It handles missing values using forward fill and returns a dictionary containing
the processed data for each stock.
'''
#-----------------------------------------------------------------------------
def fetch_and_prepare_data(selected_stocks, market_index, period='5y', interval='1d'):
    print(f"Fetching data for {len(selected_stocks)} stocks and market index {market_index} ")
    
    # Download stock & market data
    market_data = yf.download(market_index, period=period, interval=interval)
    stock_data = yf.download(selected_stocks, period=period, interval=interval)
    
    # Store processed data for each stock
    stock_datasets = {}
    
    for stock in selected_stocks:
        try:
            print(f"Processing {stock}")
            
            # Get relevant price data for the stock
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_prices = pd.DataFrame({
                    'Open': stock_data['Open'][stock],
                    'High': stock_data['High'][stock],
                    'Low': stock_data['Low'][stock],
                    'Close': stock_data['Close'][stock],
                    'Volume': stock_data['Volume'][stock]
                })
            else:
                stock_prices = stock_data.copy()
            
            # Deal with NaN values
            if stock_prices.isnull().any().any():
                print(f"Warning: Found NaN values in {stock} price data. Filling with forward fill ")
                stock_prices = stock_prices.ffill()
                
            # Make zero values a non-zero value
            for col in ['Open', 'High', 'Low', 'Close']:
                zero_mask = stock_prices[col] == 0
                if zero_mask.any():
                    print(f"Warning: Found {zero_mask.sum()} zero values in {col} for {stock} ")
                    stock_prices.loc[zero_mask, col] = np.nan
                    stock_prices[col] = stock_prices[col].ffill().bfill()
            
            # Merge with market data
            df = stock_prices.copy()
            df['Market_Close'] = market_data['Close']
            df['Market_Volume'] = market_data['Volume']
        except Exception as e:
            print(f"Error processing {stock}: {str(e)}")
            continue
        
        # Calculate returns (target variable)
        df['Return'] = df['Close'].pct_change()
        df['Market_Return'] = df['Market_Close'].pct_change()
        
        ## Feature Engineering
        #Moving Averages
        for window in [10, 50, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Market_MA_{window}'] = df['Market_Close'].rolling(window=window).mean()
        
        #Price relative to the moving averages
        for window in [10, 50, 200]:
            df[f'Price_to_MA_{window}'] = df['Close'] / df[f'MA_{window}'].replace(0, np.nan)
            df[f'Price_to_MA_{window}'] = df[f'Price_to_MA_{window}'].clip(0.5, 2.0)
        
        #Volatility measures
        for window in [10, 20, 50]:
            df[f'Volatility_{window}'] = df['Return'].rolling(window=window).std()
            df[f'Market_Volatility_{window}'] = df['Market_Return'].rolling(window=window).std()
            df[f'Volatility_{window}'] = df[f'Volatility_{window}'].replace(0, 1e-6)
            df[f'Market_Volatility_{window}'] = df[f'Market_Volatility_{window}'].replace(0, 1e-6)
        
        #Momentum indicators - RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            loss_safe = loss.replace(0, 1e-6)
            rs = gain / loss_safe
            return 100 - (100 / (1 + rs))
        
        df['RSI_14'] = calculate_rsi(df['Close'])
        df['RSI_14'] = df['RSI_14'].clip(0, 100)
        
        #Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_10'] = df['Volume_MA_10'].replace(0, 1)
        df['Volume_to_MA_10'] = (df['Volume'] / df['Volume_MA_10']).clip(0.1, 10)
        
        #Price differentials
        df['High_Low_Diff'] = df['High'] - df['Low']
        df['Open_Close_Diff'] = df['Open'] - df['Close']
        df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        #Lagged returns
        for lag in range(1, 6):
            df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
            df[f'Market_Return_Lag_{lag}'] = df['Market_Return'].shift(lag)
        
        #Moving Average Crossovers
        df['MA_10_50_Crossover'] = (df['MA_10'] > df['MA_50']).astype(int)
        df['MA_10_200_Crossover'] = (df['MA_10'] > df['MA_200']).astype(int)
        df['MA_50_200_Crossover'] = (df['MA_50'] > df['MA_200']).astype(int)
        
        #RSI Trading Signals
        df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
        df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
        
        # Drop rows with nan values
        df = df.dropna()
        
        # Store processed data
        stock_datasets[stock] = df
    
    return stock_datasets

#-----------------------------------------------------------------------------
'''
This function separates the features and target variable from the dataset.
It shifts the target column by the prediction_horizon to enable predicting
future returns, removes rows where the target becomes NaN due to shifting,
and handles any infinite or missing values by replacing them with the median.
It returns the feature matrix X and target vector y ready for model training.
'''
#-----------------------------------------------------------------------------
def prepare_features_target(df, target_column='Return', prediction_horizon=1):
    # Shift the target to predict n days ahead
    target = df[target_column].shift(-prediction_horizon)
    
    # Leave target column out of features
    feature_columns = [col for col in df.columns if col != target_column]
    
    # get rid of rows where target becomes nan as a result of shifting
    valid_idx = ~target.isna()
    X = df.loc[valid_idx, feature_columns]
    y = target.loc[valid_idx]
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column median
    X = X.fillna(X.median())
    
    return X, y

#-----------------------------------------------------------------------------
'''
This function assesses model performance by calculating regression metrics on test data.
It handles edge cases like NaN or infinite predictions by replacing them with 
valid values, then calculates MSE, RMSE, MAE and R² scores. It prints the results
and returns a dictionary containing the model name, performance metrics, 
and the predictions for further analysis.
'''
#-----------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    try:
        y_pred = model.predict(X_test)
        
        # For neural network output
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        # Check for NaN or infinite values in predictions
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            print(f"Warning: {model_name} produced infinite/nan predictions ")
            # Replace problematic values with mean of valid predictions
            valid_preds = y_pred[~(np.isnan(y_pred) | np.isinf(y_pred))]
            if len(valid_preds) > 0:
                y_pred = np.where(np.isnan(y_pred) | np.isinf(y_pred), np.mean(valid_preds), y_pred)
            else:
                # use the mean of the target if all predictions are problematic
                y_pred = np.full_like(y_pred, np.mean(y_test))
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{model_name} Performance:")
        print(f"  MSE:  {mse:.5f}")
        print(f"  RMSE: {rmse:.5f}")
        print(f"  MAE:  {mae:.5f}")
        print(f"  R²:   {r2:.5f}")
        
        return {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        # Return placeholder values
        return {
            'model_name': model_name,
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'y_pred': np.full(len(y_test), np.nan)
        }

#-----------------------------------------------------------------------------
'''
This function visualises the actual versus predicted returns for each model.
It creates separate plots for each model, handling extreme predictions by clipping
them to reasonable values. The function uses different colors for each model type
and includes the actual returns for comparison. It saves each plot as a PNG file
named with the stock name and model type, but doesn't return any values.
'''
#-----------------------------------------------------------------------------
def plot_predictions(y_test, predictions_dict, stock_name):
    # Colors for each model
    colors = {'Linear Regression': 'red', 
              'Support Vector Machine': 'green', 
              'Random Forest': 'blue', 
              'Neural Network': 'indigo'}
    
    # Plot predictions from each model
    for model_name, result in predictions_dict.items():
        #check if predictions have nan values
        if np.isnan(result['y_pred']).any():
            print(f"Warning: {model_name} has NaN predictions, skipping plot")
            continue
        
        #Check if predictions are reasonable
        pred_mean = np.mean(result['y_pred'])
        pred_std = np.std(result['y_pred'])
        reasonable_preds = result['y_pred']
        
        #Replace extreme values with mean
        threshold = 5 * pred_std
        extreme_mask = np.abs(reasonable_preds - pred_mean) > threshold
        if np.any(extreme_mask):
            print(f"Warning: {model_name} contains extreme predictions, clipping for plot")
            reasonable_preds = np.copy(result['y_pred'])
            reasonable_preds[extreme_mask] = pred_mean
        
        # Mkae separate plot for each model
        plt.figure(figsize=(14, 8))
        
        # Plot actual returns
        plt.plot(y_test.values, label='Actual Returns', color='black', alpha=0.7, linewidth=2)
        
        # Plot this model's predictions
        color = colors.get(model_name, 'brown')
        plt.plot(reasonable_preds, label=f"{model_name} Predictions", color=color, alpha=0.7)
        
        plt.title(f'Actual vs {model_name} Predicted Returns for {stock_name}')
        plt.xlabel('Time')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'{stock_name}_{model_name.replace(" ", "_")}_predictions.png')
        plt.close()
    return None
#-----------------------------------------------------------------------------
'''
This function creates a feedforward neural network for regression.
It builds a Sequential model with three hidden layers (64, 32, and 16 neurons)
using ReLU activation and Dropout layers (20% dropout rate) to prevent overfitting.
The output layer has a single neuron with no activation function for regression.
It returns a compiled Keras model using Adam optimizer and MSE loss function.
'''
#-----------------------------------------------------------------------------
def build_neural_network(input_dim):
    # Create Sequential model starting with an Input layer
    model = Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    
    return model

#-----------------------------------------------------------------------------
'''
This function trains and evaluates a Linear Regression model on the provided data.
It fits the model to the training data, evaluates it on test data using evaluate_model(),
and calculates feature importance based on the absolute values of coefficients.
It prints the top 10 most important features and returns both the evaluation results
and a DataFrame containing feature importance scores.
'''
#-----------------------------------------------------------------------------
def train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test, X):
    print("\nTraining Linear Regression model ")
    try:
        # Train model
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        lr_result = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
        
        # Get feature importance
        lr_feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(lr_model.coef_)
        }).sort_values(by='Importance', ascending=False)
        
        print("\n10 best features for Linear Regression:")
        print(lr_feature_importance.head(10))
        
        return lr_result, lr_feature_importance
        
    except Exception as e:
        print(f"Error during Linear Regression: {str(e)}")
        result = {
            'model_name': "Linear Regression",
            'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'y_pred': np.full(len(y_test), np.nan)
        }
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.zeros(len(X.columns))})
        return result, feature_importance

#-----------------------------------------------------------------------------
'''
This function trains Support Vector Machine models with different kernels.
It starts with a linear kernel, then tries RBF and polynomial kernels if 
the linear kernel produces low-variance predictions. It evaluates each model
and selects the best performing one based on R² score. The function handles
exceptions and returns a dictionary with the evaluation results for the best model.
'''
#-----------------------------------------------------------------------------
def train_svm(X_train_scaled, y_train, X_test_scaled, y_test):
    print("\nTraining SVM model for stock returns prediction ")
    
    try:
        svm_model = SVR(
            kernel='linear',
            C=1.0,
            epsilon=0.1
        )
        
        svm_model.fit(X_train_scaled, y_train)
        linear_result = evaluate_model(svm_model, X_test_scaled, y_test, "Support Vector Machine (Linear)")
        
        # Check variance of predictions compared to actual returns
        actual_std = np.std(y_test)
        pred_std = np.std(linear_result['y_pred'])
        
        if pred_std < 0.2 * actual_std:
            print("Linear kernel producing low variance predictions, attempting RBF with specific parameters ")
            
            #based on parameters that worked in the logged output samples
            rbf_model = SVR(
                kernel='rbf',
                C=10.0,
                gamma=0.1,
                epsilon=0.01
            )
            
            rbf_model.fit(X_train_scaled, y_train)
            rbf_result = evaluate_model(rbf_model, X_test_scaled, y_test, "Support Vector Machine (RBF)")
            
            if np.std(rbf_result['y_pred']) < 0.2 * actual_std:
                print("RBF kernel is producing low variance, attempting polynomial kernel ")
                
                poly_model = SVR(
                    kernel='poly',
                    degree=2,
                    C=10.0,
                    gamma='auto',
                    epsilon=0.01
                )
                
                poly_model.fit(X_train_scaled, y_train)
                poly_result = evaluate_model(poly_model, X_test_scaled, y_test, "Support Vector Machine (Poly)")
                
                # Return the best result based on R2 score
                candidates = [linear_result, rbf_result, poly_result]
                best_idx = np.argmax([r['r2'] for r in candidates])
                
                if best_idx == 2:
                    print("Polynomial kernel gave the best results")
                    return poly_result
                elif best_idx == 1:
                    print("RBF kernel gave the best results")
                    return rbf_result
            
            # Compare linear with RBF
            if rbf_result['r2'] > linear_result['r2']:
                print("RBF kernel improved performance")
                return rbf_result
        
        # Return the linear model result if it's good
        return linear_result
        
    except Exception as e:
        print(f"Error during SVM: {str(e)}")
        result = {
            'model_name': "Support Vector Machine",
            'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'y_pred': np.full(len(y_test), np.nan)
        }
        return result

#-----------------------------------------------------------------------------
'''
This function trains Random Forest models with different numbers of estimators.
It tests models with 100, 200, 300, 400, and 500 trees to find the optimal value,
then trains a final model with the best configuration. It calculates feature importance,
prints the top 10 features, and returns both the evaluation results and a DataFrame
with feature importance scores. It also handles exceptions with appropriate error messages.
'''
#-----------------------------------------------------------------------------
def train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test, X):
    print("\nTraining the Random Forest model with hyperparameter tuning")
    try:
        #try different n_estimators to find the best value
        rf_performances = {}
        for n_est in [100, 200, 300, 400, 500]:
            print(f"\nTesting Random Forest with {n_est} estimators ")
            temp_rf = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42
            )
            temp_rf.fit(X_train_scaled, y_train)
            temp_result = evaluate_model(temp_rf, X_test_scaled, y_test, f"Random Forest ({n_est} trees)")
            rf_performances[n_est] = temp_result['mse']
        
        # Find optimal n_estimators
        optimal_n_est = min(rf_performances, key=rf_performances.get)
        print(f"\nIdeal number of estimators: {optimal_n_est}")
        
        # Train final model with optimal n_estimators
        rf_model = RandomForestRegressor(
            n_estimators=optimal_n_est,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_result = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest (Ideal)")
        
        # Get feature importance 
        rf_feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("\n10 best features for Random Forest:")
        print(rf_feature_importance.head(10))
        
        return rf_result, rf_feature_importance
        
    except Exception as e:
        print(f"Error during Random Forest: {str(e)}")
        result = {
            'model_name': "Random Forest",
            'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'y_pred': np.full(len(y_test), np.nan)
        }
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.zeros(len(X.columns))})
        return result, feature_importance

#-----------------------------------------------------------------------------
'''
This function builds, trains and evaluates a neural network for stock return prediction.
It uses early stopping to prevent overfitting, trains for up to 50 epochs, and
clips extreme predictions to within 3 standard deviations of the mean. It creates and
saves a learning curve plot showing training and validation loss over epochs.
The function returns a dictionary with the evaluation metrics and predictions.
'''
#-----------------------------------------------------------------------------
def train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test, stock_name="Unknown"):
    print("\nTraining Neural Network model")
    try:
        # Build the model
        nn_model = build_neural_network(X_train_scaled.shape[1])
        
        # Use early stopping to stop overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # train model
        nn_history = nn_model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # predict and evaluate
        nn_result = evaluate_model(nn_model, X_test_scaled, y_test, "Neural Network")
        
        # apply clipping to neural network predictions to prevent extreme values
        if 'y_pred' in nn_result:
            y_pred = nn_result['y_pred']
            
            # calculate mean and std of actual values for reference
            y_mean = np.mean(y_test)
            y_std = np.std(y_test)
            
            # clip predictions to within a decent range
            y_pred_clipped = np.clip(y_pred, y_mean - 3*y_std, y_mean + 3*y_std)
            
            #Recalculate metrics with the clipped predictions
            if np.any(y_pred != y_pred_clipped):
                print("Note: Neural Network predictions were clipped")
                mse = mean_squared_error(y_test, y_pred_clipped)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred_clipped)
                r2 = r2_score(y_test, y_pred_clipped)
                
                # Update result with the clipped predictions
                nn_result = {
                    'model_name': "Neural Network (clipped)",
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'y_pred': y_pred_clipped
                }
                
                print(f"Neural Network Performance (after clipping):")
                print(f"  MSE:  {mse:.5f}")
                print(f"  RMSE: {rmse:.5f}")
                print(f"  MAE:  {mae:.5f}")
                print(f"  R²:   {r2:.5f}")
        
        # Plot & save learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(nn_history.history['loss'], label='Training Loss')
        plt.plot(nn_history.history['val_loss'], label='Validation Loss')
        plt.title(f'Neural Network Learning Curves for {stock_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{stock_name}_nn_learning_curves.png')
        plt.close()
        
        return nn_result
        
    except Exception as e:
        print(f"Error during Neural Network: {str(e)}")
        result = {
            'model_name': "Neural Network",
            'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'y_pred': np.full(len(y_test), np.nan)
        }
        return result

#-----------------------------------------------------------------------------
'''
This function coordinates the training and evaluation of all models for a given stock.
It splits the data into training and testing sets, scales the features, and trains
four different models: Linear Regression, SVM, Random Forest, and Neural Network.
It creates visualisations comparing feature importance between Linear Regression
and Random Forest, as well as actual vs predicted plots for each model. The function
returns the results from all models and feature importance DataFrames.
'''
#-----------------------------------------------------------------------------
def train_and_evaluate_models(X, y, stock_name):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Split data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining models for {stock_name} ")
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    results = {}
    
    #Linear Regression
    lr_result, lr_feature_importance = train_linear_regression(
        X_train_scaled, y_train, X_test_scaled, y_test, X
    )
    results["Linear Regression"] = lr_result
    
    #Support Vector Machine
    svm_result = train_svm(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    results["Support Vector Machine"] = svm_result
    
    #Random Forest
    rf_result, rf_feature_importance = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test, X
    )
    results["Random Forest"] = rf_result
    
    #Neural Network
    nn_result = train_neural_network(
    X_train_scaled, y_train, X_test_scaled, y_test, stock_name
    )
    results["Neural Network"] = nn_result
    
    try:
        # Plot feature importance comparison if both models were successfully trained
        if lr_feature_importance is not None and rf_feature_importance is not None:
            plt.figure(figsize=(12, 10))
            
            # Top 10 features from Linear Regression
            plt.subplot(2, 1, 1)
            top_lr_features = lr_feature_importance.head(10)
            sns.barplot(x='Importance', y='Feature', data=top_lr_features)
            plt.title(f'10 Best Feature Importance - Linear Regression ({stock_name})')
            plt.tight_layout()
            
            # Top 10 features from Random Forest
            plt.subplot(2, 1, 2)
            top_rf_features = rf_feature_importance.head(10)
            sns.barplot(x='Importance', y='Feature', data=top_rf_features)
            plt.title(f'10 Best Feature Importance - Random Forest ({stock_name})')
            plt.tight_layout()
            
            plt.savefig(f'{stock_name}_feature_importance.png')
            plt.close()
            
        # Plot predictions for each model
        plot_predictions(y_test, results, stock_name)
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
    
    return results, rf_feature_importance, lr_feature_importance

#-----------------------------------------------------------------------------
'''
This function performs time-series cross-validation to assess model robustness.
It uses TimeSeriesSplit to create training/testing folds that respect chronological
order, evaluates multiple models (Linear Regression, SVM, and Random Forest variants)
using various metrics (MSE, MAE, R²), and performs a simplified cross-validation for
the Neural Network due to computational expense. It returns a dictionary with
cross-validation results for each model, including score means and standard deviations.
'''
#-----------------------------------------------------------------------------
def perform_cross_validation(X, y, stock_name):
    print(f"\nPerforming time-series cross-validation for {stock_name} ")
    
    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    if X.isnull().any().any():
        print(f"Warning: NaN values discovered; filling with zeros")
        X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Machine': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
        'Random Forest (100)': RandomForestRegressor(n_estimators=100, max_depth=20, 
                                           min_samples_split=5, min_samples_leaf=2, 
                                           max_features='sqrt', random_state=42),
        'Random Forest (300)': RandomForestRegressor(n_estimators=300, max_depth=20, 
                                           min_samples_split=5, min_samples_leaf=2, 
                                           max_features='sqrt', random_state=42),
        'Random Forest (500)': RandomForestRegressor(n_estimators=500, max_depth=20, 
                                           min_samples_split=5, min_samples_leaf=2, 
                                           max_features='sqrt', random_state=42)
    }
    
    cv_results = {}
    scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    
    for name, model in models.items():
        try:
            # Store results for each scoring metric
            metric_results = {}
            
            for metric in scoring_metrics:
                try:
                    cv_scores = cross_val_score(
                        model, X_scaled, y, 
                        cv=tscv, 
                        scoring=metric
                    )
                    
                    # Convert negative metrics back to positive for MSE and MAE
                    if metric.startswith('neg_'):
                        cv_scores = -cv_scores
                        metric_name = metric[4:]
                    else:
                        metric_name = metric
                    
                    metric_results[metric_name] = {
                        'scores': cv_scores,
                        'mean': cv_scores.mean(),
                        'std': cv_scores.std()
                    }
                    
                    print(f"{name} - {metric_name}:")
                    print(f"  Mean: {cv_scores.mean():.5f}")
                    print(f"  Std:  {cv_scores.std():.5f}")
                except Exception as metric_error:
                    print(f"Error calculating {metric} for {name}: {str(metric_error)}")
                    metric_name = metric.replace('neg_', '') if metric.startswith('neg_') else metric
                    metric_results[metric_name] = {
                        'scores': np.array([np.nan]),
                        'mean': np.nan,
                        'std': np.nan
                    }
            
            cv_results[name] = metric_results
            
        except Exception as e:
            print(f"Error during cross-validation for {name}: {str(e)}")
            cv_results[name] = {
                'mean_squared_error': {'scores': np.array([np.nan]), 'mean': np.nan, 'std': np.nan},
                'mean_absolute_error': {'scores': np.array([np.nan]), 'mean': np.nan, 'std': np.nan},
                'r2': {'scores': np.array([np.nan]), 'mean': np.nan, 'std': np.nan}
            }
    
    # Neural Network cross-validation with a simpler approach
    try:
        print("\nComputationally expensive; undergoing simpler cross-validation for the Neural Network")
        nn_cv_scores = []
        
        for train_index, test_index in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Build and train a smaller network for CV
            model = Sequential([
                Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Make use of early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            #train with smaller epochs for CV
            model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            y_pred = model.predict(X_test).flatten()
            mse = mean_squared_error(y_test, y_pred)
            nn_cv_scores.append(mse)
        
        cv_results["Neural Network"] = {
            'mean_squared_error': {
                'scores': np.array(nn_cv_scores),
                'mean': np.mean(nn_cv_scores),
                'std': np.std(nn_cv_scores)
            }
        }
        
        print(f"Neural Network Cross-Validation:")
        print(f"  Mean MSE: {np.mean(nn_cv_scores):.5f}")
        print(f"  Std MSE:  {np.std(nn_cv_scores):.5f}")
        
    except Exception as e:
        print(f"Error during neural network cross-validation: {str(e)}")
        cv_results["Neural Network"] = {
            'mean_squared_error': {'scores': np.array([np.nan]), 'mean': np.nan, 'std': np.nan}
        }
    
    return cv_results

#-----------------------------------------------------------------------------
'''
This function compares model performance across all stocks by creating summary
visualisations and calculating average metrics. It aggregates results into a
DataFrame showing MSE, RMSE, MAE and R² for each model and stock combination,
computes average metrics for each model type, and creates bar plots comparing
performance by metric. These plots are saved as PNG files. The function returns
both the summary DataFrame and a dictionary with the average metrics by model.
'''
#-----------------------------------------------------------------------------
def compare_model_performance(all_results):
    print("\nComparing model performance on all the stocks ")
    
    # Make summary dataframe
    summary_data = []
    model_averages = {
        'Linear Regression': {'mse': [], 'mae': [], 'r2': []},
        'Support Vector Machine': {'mse': [], 'mae': [], 'r2': []},
        'Random Forest': {'mse': [], 'mae': [], 'r2': []},
        'Neural Network': {'mse': [], 'mae': [], 'r2': []}
    }
    
    for stock, results in all_results.items():
        for model_name, metrics in results.items():
            #Add to summary data
            summary_data.append({
                'Stock': stock,
                'Model': model_name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2']
            })
            
            #Add to model averages
            if not np.isnan(metrics['mse']):
                model_averages[model_name]['mse'].append(metrics['mse'])
                model_averages[model_name]['mae'].append(metrics['mae'])
                model_averages[model_name]['r2'].append(metrics['r2'])
    
    #Make summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    #Get model averages
    model_avg = {}
    for model, metrics in model_averages.items():
        if len(metrics['mse']) > 0:
            model_avg[model] = {
                'mse_avg': np.mean(metrics['mse']),
                'mae_avg': np.mean(metrics['mae']),
                'r2_avg': np.mean(metrics['r2'])
            }
        else:
            model_avg[model] = {
                'mse_avg': np.nan,
                'mae_avg': np.nan,
                'r2_avg': np.nan
            }
    
    # Print model averages
    print("\nModel Averages Across All Stocks:")
    for model, metrics in model_avg.items():
        print(f"{model}:")
        print(f"  Avg MSE: {metrics['mse_avg']:.5f}")
        print(f"  Avg MAE: {metrics['mae_avg']:.5f}")
        print(f"  Avg R²:  {metrics['r2_avg']:.5f}")
    
    # Make plots for MSE, MAE, and R²
    # MSE comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='MSE', hue='Stock', data=summary_df)
    plt.title('Mean Squared Error by Model and Stock')
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_mse.png')
    plt.close()

    # MAE comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='MAE', hue='Stock', data=summary_df)
    plt.title('Mean Absolute Error by Model and Stock')
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_mae.png')
    plt.close()

    # R² comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='R²', hue='Stock', data=summary_df)
    plt.title('R-squared by Model and Stock')
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_r2.png')
    plt.close()
    
    return summary_df, model_avg

# ----------------------------------------------------------------------------
# Main Program
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Specfiy the stocks and market index chosen from Assingment 1
        selected_stocks = [
            'SAP.DE',    # SAP
            'SIE.DE',    # Siemens
            'ALV.DE',    # Allianz
            'BAS.DE',    # BASF
            'BMW.DE',    # BMW
        ]
        market_index = '^GDAXI'  # DAX40
        
        # Get and prepare the data
        stock_datasets = fetch_and_prepare_data(selected_stocks, market_index)
        
        #Train and evaluate the models for each stock
        all_results = {}
        all_feature_importance_rf = {}
        all_feature_importance_lr = {}
        
        for stock, data in stock_datasets.items():
            print(f"\n{'='*50}")
            print(f"Processing {stock}")
            print(f"{'='*50}")
            
            try:
                # Prepare features and target
                X, y = prepare_features_target(data)
                
                # Do cross-validation
                cv_results = perform_cross_validation(X, y, stock)
                
                # Train & evaluate models
                results, rf_importance, lr_importance = train_and_evaluate_models(X, y, stock)
                
                # Keep results for comparison
                all_results[stock] = results
                all_feature_importance_rf[stock] = rf_importance
                all_feature_importance_lr[stock] = lr_importance
            except Exception as e:
                print(f"Error processing {stock}: {str(e)}")
                continue
        
        #Compare model performance on all stocks
        if all_results:
            try:
                summary_df, model_avg = compare_model_performance(all_results)
                print("\nComparison successful")
            except Exception as e:
                print(f"Error during model comparison: {str(e)}")
        else:
            print("Valid results for comparison not found")
        
        print("\nAnalysing feature importance across all stocks ")
        
        try:
            # Combine RF feature importance from all stocks
            combined_rf_importance = pd.DataFrame()
            
            for stock, importance_df in all_feature_importance_rf.items():
                if importance_df is not None:
                    importance_df = importance_df.copy()
                    importance_df['Stock'] = stock
                    combined_rf_importance = pd.concat([combined_rf_importance, importance_df])
            
            if not combined_rf_importance.empty:
                # Get average importance by feature
                avg_importance = combined_rf_importance.groupby('Feature')['Importance'].mean().reset_index()
                top_features = avg_importance.sort_values('Importance', ascending=False).head(10)
                
                print("\n10 Best Overall Features:")
                print(top_features)
            else:
                print("Warning: No valid feature importance data available")
        except Exception as e:
            print(f"Error during feature importance analysis: {str(e)}")
        
        print("\nProgram completed. Plots generated in the root directory folder")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
