import joblib

# Replace 'your_file.pkl' with the path to your .pkl file
data = joblib.load('BTCUSDm_model.pkl')
print(data)

# Access the XGBoost model and the scaler
xgb_model = data['model']
scaler = data['scaler']

# Now you can use the scaler and model for predictions
print(xgb_model)   # Print the XGBoost model
print(scaler)      # Print the StandardScaler

# # Example usage:
# # Assume you have a dataset `X_test` that needs to be scaled before predictions
# X_scaled = scaler.transform(X_test)
# predictions = xgb_model.predict(X_scaled)

# print(predictions)