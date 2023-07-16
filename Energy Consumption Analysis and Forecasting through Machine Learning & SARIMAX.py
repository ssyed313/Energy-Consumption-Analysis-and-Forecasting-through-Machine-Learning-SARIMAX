import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the load time series and encoded time covariates dataset
load_data = pd.read_csv("/content/series_2018_2020.csv")
covariates_data = pd.read_csv("/content/time_covariates_2018_2020.csv")

# Merge load data and covariates data
merged_data = pd.merge(load_data, covariates_data, on="Data e Hora")

# Convert the date strings in the merged_data to datetime objects
merged_data["Data e Hora"] = pd.to_datetime(merged_data["Data e Hora"], format="%d/%m/%Y %H:%M")

# Separate features and target variables
X = merged_data[['year', 'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos', 'hour_sin', 'hour_cos',
                 'minute_sin', 'minute_cos', 'dayofweek_sin', 'dayofweek_cos', 'weekofyear_sin', 'weekofyear_cos',
                 'holidays']]
y = merged_data["Consumo"]

# Preprocess the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)

# Perform dimensionality reduction using PCA
pca = PCA(n_components=14)
X_pca = pca.fit_transform(scaled_data)

# Print explained variance ratio of PCA
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Split the data into training and testing sets
train_size = int(0.8 * len(merged_data))
X_train = X_pca[:train_size]
y_train = y[:train_size]
X_test = X_pca[train_size:, :]
y_test = y[train_size:]

# Define and train machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=0.1),  # Adjust alpha value if needed
    "Lasso Regression": Lasso(alpha=0.1),  # Adjust alpha value if needed
    "Support Vector Regression": SVR(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
}

# Define parameter grids for GridSearchCV
param_grids = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": [0.01, 0.1, 1.0]},  # Adjust alpha values if needed
    "Lasso Regression": {"alpha": [0.01, 0.1, 1.0]},  # Adjust alpha values if needed
    "Support Vector Regression": {"C": [1.0, 10.0, 100.0], "kernel": ["linear", "rbf"]},
    "Random Forest": {"n_estimators": [100, 200, 300]},
    "Gradient Boosting": {"n_estimators": [100, 200, 300]},
}

best_models = {}

# Perform grid search for each model
for model_name, model in models.items():
    print("Training", model_name)
    grid_search = GridSearchCV(model, param_grid=param_grids[model_name], scoring="neg_mean_squared_error",
                               cv=TimeSeriesSplit(n_splits=3))
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    print()

# Evaluate the models on the test set and plot the forecast
for model_name, model in best_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(model_name, "MSE:", mse)
    print(model_name, "MAE:", mae)
    print()

    plt.plot(y_test.index, y_test.values, label="True")
    plt.plot(y_test.index, y_pred, label=model_name)
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.title("Model Forecast on Test Set - " + model_name)
    plt.legend()
    plt.show()

# Perform cross-validation for each model
for model_name, model in best_models.items():
    scores = cross_val_score(model, X_pca, y, scoring='neg_mean_squared_error', cv=TimeSeriesSplit(n_splits=3))
    mse_scores = -scores
    mae_scores = np.sqrt(mse_scores)
    print(model_name, "Cross-Validation MSE:", mse_scores)
    print(model_name, "Cross-Validation MAE:", mae_scores)
    print("Mean CV MSE:", np.mean(mse_scores))
    print("Mean CV MAE:", np.mean(mae_scores))
    print()

# Train SARIMAX model
sarimax_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
sarimax_model_fit = sarimax_model.fit()

# Forecast using SARIMAX model
sarimax_forecast = sarimax_model_fit.forecast(steps=len(y_test))

# Evaluate SARIMAX model
sarimax_mse = mean_squared_error(y_test, sarimax_forecast)
sarimax_mae = mean_absolute_error(y_test, sarimax_forecast)
print("SARIMAX MSE:", sarimax_mse)
print("SARIMAX MAE:", sarimax_mae)

# Plot SARIMAX forecast
plt.plot(y_test.index, y_test.values, label="True")
plt.plot(y_test.index, sarimax_forecast, label="SARIMAX")
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("SARIMAX Forecast")
plt.legend()
plt.show()

# Find the best model based on the forecast
best_model_name = None
best_mse = float("inf")
for model_name, model in best_models.items():
    forecast = model.predict(X_test)
    mse = mean_squared_error(y_test, forecast)
    if mse < best_mse:
        best_mse = mse
        best_model_name = model_name

# Plot the forecast of the best model and SARIMAX
best_model = best_models[best_model_name]
forecast = best_model.predict(X_test)

# Create a new DataFrame with the timestamps and corresponding forecast values
forecast_df = pd.DataFrame({'Timestamp': y_test.index, 'Forecast': forecast})

# Plot the forecast using the formatted timestamps on the x-axis
plt.plot(forecast_df['Timestamp'], y_test.values, label="True")
plt.plot(forecast_df['Timestamp'], forecast, label=best_model_name)
plt.plot(forecast_df['Timestamp'], sarimax_forecast, label="SARIMAX")
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Best Model vs. SARIMAX Forecast")
plt.legend()
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()