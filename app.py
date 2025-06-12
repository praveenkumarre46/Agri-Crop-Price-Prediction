from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model and scaler
model = joblib.load('best_crop_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the datasets
df1 = pd.read_csv('groundnut3.csv')
df2 = pd.read_csv('paddy3.csv')

# Add Crop Name to the datasets
df1['Crop Name'] = 'Groundnut'
df2['Crop Name'] = 'Paddy'

# Combine the datasets
combined_data = pd.concat([df1, df2], ignore_index=True)

# Encode categorical variables
combined_data = pd.get_dummies(combined_data, columns=['Month', 'Season', 'Crop Name'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        app.logger.info(f"Received data: {data}")

        # Extract input features
        year = data['Year']
        month = data.get('Month')
        season = data.get('Season')
        crop_name = data['Crop Name']

        # Create a dataframe for the input data
        input_data = {'Year': [year]}
        
        # One-hot encode the categorical variables
        if month:
            month_col = f'Month_{month.capitalize()}'
            input_data[month_col] = [1]
        elif season:
            season_col = f'Season_{season}'
            input_data[season_col] = [1]

        crop_col = f'Crop Name_{crop_name}'
        input_data[crop_col] = [1]

        # Convert input_data to DataFrame
        input_data_df = pd.DataFrame(input_data)

        # Ensure all expected columns are present
        for col in scaler.get_feature_names_out():
            if col not in input_data_df.columns:
                input_data_df[col] = 0

        # Reorder columns to match the order used during training
        input_data_df = input_data_df[scaler.get_feature_names_out()]

        # Scale the input data
        scaled_data = scaler.transform(input_data_df)
        prediction = model.predict(scaled_data)
        
        # Convert the predicted value to a standard Python type and format to 2 decimal places
        predicted_price = round(float(prediction[0]), 2)
        
        return jsonify({'predicted_price': f'{predicted_price} Rupees per Quintal'})
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/data')
def data():
    return combined_data.to_html()

if __name__ == '__main__':
    app.run(debug=True)
