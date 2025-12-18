# Agri Crop Price Prediction

## Overview
This project predicts agricultural crop prices using historical market data.
It uses machine learning models to learn price patterns and serves predictions through a Flask application.
The goal is to help farmers and stakeholders make better selling and planning decisions.

## Machine Learning Approach

The project uses a single machine learning regression model to predict crop prices.
Historical crop market data is used directly without feature scaling.

Several models were experimented with, and **XGBoost** provided the best performance.
The trained model is saved and later used by the Flask application to generate price predictions.
