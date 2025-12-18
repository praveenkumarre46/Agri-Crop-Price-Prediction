from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

# Train and evaluate each model
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
