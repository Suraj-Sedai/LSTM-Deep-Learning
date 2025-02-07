import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf  # Fetching stock data

# Step 1: Load Data (Downloading Apple's stock prices)
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Step 2: Preprocess Data
close_prices = stock_data['Close'].values  # Use closing prices
scaler = MinMaxScaler(feature_range=(0,1))  # Normalize data
close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1,1))

# Step 3: Create Training Data
X_train, y_train = [], []
time_steps = 60  # Using 60 past days to predict next day

for i in range(len(close_prices_scaled) - time_steps):
    X_train.append(close_prices_scaled[i:i+time_steps])  # 60 past days
    y_train.append(close_prices_scaled[i+time_steps])  # Predict next day

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Step 4: Build LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')

# Step 5: Train Model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Step 6: Predict the Next Day's Stock Price
last_60_days = close_prices_scaled[-time_steps:].reshape(1, time_steps, 1)
predicted_price = model.predict(last_60_days)

# Convert back to original scale
predicted_price = scaler.inverse_transform(predicted_price)
print(f"Predicted Stock Price for Next Day: {predicted_price[0][0]}")

# Step 7: Visualizing
plt.plot(close_prices, label="Actual Prices")
plt.axhline(predicted_price[0][0], color='red', linestyle='dashed', label="Predicted Price")
plt.legend()
plt.show()
