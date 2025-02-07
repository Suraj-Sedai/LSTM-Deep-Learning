#Using past data for prediction
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70,80,90,100])
x, y = [], []

#using past 3 data to predict the next data
for i in range(len(data) - 4):
    x.append(data[i:i+3])
    y.append(data[i+3])

x = np.array(x).reshape((len(x), 3, 1))
y = np.array(y)

# Step 2: Build the LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Step 3: Train the Model
model.fit(x, y, epochs=1000, verbose=0)

# Step 4: Make Predictions
input_data = np.array([80, 90, 100]).reshape(1, 3, 1)
prediction = model.predict(input_data)
print(f"Predicted next number after 80, 90, 100: {prediction[0][0]}")