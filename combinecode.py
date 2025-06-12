import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

# Load the datasets
df1 = pd.read_csv('groundnut4.csv')
df2 = pd.read_csv('paddy4.csv')

# Combine the datasets
combined_data = pd.concat([df1, df2], ignore_index=True)

# Normalize month values to handle case sensitivity
combined_data['Month'] = combined_data['Month'].str.strip().str.capitalize()

# Handle missing values
combined_data.ffill(inplace=True)

# Encode categorical variables
combined_data = pd.get_dummies(combined_data, columns=['Month', 'Season', 'Crop name'])

# Ensure the target column exists before proceeding
target_column = 'Avg Market Price (Rupee/Quintal)'
if target_column not in combined_data.columns:
    raise KeyError(f"Column '{target_column}' not found in the dataset")

# Select relevant features
features = ['Year', 'Total Area Cultivated (Ha)', 'Total Production (Tonnes)', 
            'Yield (Tonnes/Ha)', 'Avg Rainfall (mm)', 'Avg Temp (deg C)', 
            'MSP (Rupee/Quintal)', 'Inflation Rate (%)', 'Fuel Prices (Rupee/L)', 
            'Export Demand (Tonnes)'] + [col for col in combined_data.columns if 'Month' in col or 'Season' in col or 'Crop name' in col]
X = combined_data[features]
y = combined_data[target_column]

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the preprocessed data, scaler, and imputer
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')

# Model Selection, Training, and Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results[name] = {"MAE": mae, "RMSE": rmse}

# Print the results
for name, metrics in results.items():
    print(f"{name} - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

# Select the best model based on RMSE
best_model_name = min(results, key=lambda x: results[x]["RMSE"])
best_model = models[best_model_name]
print(f"\nThe best model is: {best_model_name}")

# Evaluate the best model on the test set
best_model.fit(X_train_scaled, y_train)
best_predictions = best_model.predict(X_test_scaled)
best_mae = mean_absolute_error(y_test, best_predictions)
best_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))

print(f"\nBest Model - {best_model_name}:")
print(f"MAE: {best_mae:.2f}")
print(f"RMSE: {best_rmse:.2f}")

# Save the best model to a file
joblib.dump(best_model, 'best_crop_price_model.pkl')
