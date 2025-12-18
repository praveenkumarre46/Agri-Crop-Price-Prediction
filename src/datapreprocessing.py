import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load and Combine Datasets

# Load the CSV files
df1 = pd.read_csv('groundnut3.csv')
df2 = pd.read_csv('paddy3.csv')

# Combine the datasets
df = pd.concat([df1, df2], ignore_index=True)

# Drop duplicates if there are any
df.drop_duplicates(inplace=True)

# Strip leading/trailing spaces and standardize column names
df.columns = df.columns.str.strip().str.capitalize()

# Normalize month values to handle case sensitivity
df['Month'] = df['Month'].str.strip().str.capitalize()

# Display the unique values for 'Year', 'Month', and 'Season'
unique_years = df['Year'].unique()
unique_months = df['Month'].unique()
unique_seasons = df['Season'].unique()

print("Unique Years:", unique_years)
print("Unique Months:", unique_months)
print("Unique Seasons:", unique_seasons)

# Display the first few rows of the combined dataset
print("Combined Data:")
print(df.head())

# Step 2: Data Preprocessing

# Handle missing values
df.ffill(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Month', 'Season'])

# Verify the column names after encoding
print("Column names after encoding:")
print(df.columns)

# Function to filter data based on user input
def filter_data(df, year=None, month=None, crop_name=None):
    filtered_df = df
    if year:
        filtered_df = filtered_df[filtered_df['Year'] == year]
    if month:
        month_col = f'Month_{month.capitalize()}'
        if month_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[month_col] == 1]
    if crop_name:
        season_col = f'Season_{crop_name}'
        if season_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[season_col] == 1]
    return filtered_df

# Use relaxed filtering criteria to ensure sufficient data
filtered_data = filter_data(df, year=2022, month=None, crop_name=None)

if filtered_data.empty or len(filtered_data) < 10:
    raise ValueError("No suitable data found for training. Please check the dataset or adjust the criteria.")

print(f"Filtered dataset contains {len(filtered_data)} samples.")

# Split the data into features and target variable
X = filtered_data.drop('Avg market price (rupee/quintal)', axis=1)
y = filtered_data['Avg market price (rupee/quintal)']

# Step 3: Split Data into Training and Testing Sets

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the preprocessed data and the scaler
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
X_train_scaled_df.to_csv('X_train_scaled.csv', index=False)
X_test_scaled_df.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
joblib.dump(scaler, 'scaler.pkl')

# Display the preprocessed data
print("Preprocessed Data:")
print(X_train_scaled_df.head())
