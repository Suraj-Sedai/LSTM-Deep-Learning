import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Step 1: Prepare the data
data = np.array([10, 20, 30, 40, 50, 60, 70])
X = []
y = []

# Creating input-output pairs
for i in range(len(data) - 1):
    X.append(data[i])
    y.append(data[i + 1])

# Reshape the data to (samples, time_steps, features) 
X = np.array(X).reshape((len(X), 1, 1))
y = np.array(y)

# Step 2: Build the LSTM Model
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(1, 1)))  # LSTM layer
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')  # Compile the model

# Step 3: Train the Model
model.fit(X, y, epochs=1000, verbose=0)

# Step 4: Make Predictions
prediction = model.predict(np.array([[70]]).reshape(1, 1, 1))
print(f"Predicted next number after 70: {prediction[0][0]}")
