#many to many LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# New dataset
data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
X, y = [], []

# Using 5 past values to predict next 5 values
for i in range(len(data) - 10 + 1):  
    X.append(data[i:i+5])  # 5 past values
    y.append(data[i+5:i+10])  # Predict next 5 values

X = np.array(X).reshape((len(X), 5, 1))  # Reshape for LSTM
y = np.array(y)  # Multiple output values

# Build the Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(5, 1)))  # Using 5 past values
model.add(Dense(5))  # Predicting 5 values
model.compile(optimizer='adam', loss='mse')

# Train the Model
model.fit(X, y, epochs=1000, verbose=0)

# Predict Next 5 Numbers After [10, 12, 14, 16, 18]
input_data = np.array([[12, 14, 16, 18, 20]]).reshape(1, 5, 1)
prediction = model.predict(input_data)
print(f"Predicted next 5 numbers: {prediction[0]}")
