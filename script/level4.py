#multi step prediction
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Step 1: Prepare the data
data = np.array([10, 20, 30, 40, 50, 60, 70])
X = []
y = []

for i in range(len(data) - 1):
    X.append(data[i])
    y.append(data[i + 1])

X = np.array(X).reshape((len(X), 1, 1))
y = np.array(y)

#step 2:build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#step 3: Train the model
model.fit(X, y, epochs=500, verbose=0)

#future predictions
future_prediations = []
current_input = np.array([[70]]).reshape(1, 1, 1)

for i in range(3):
    future_pred = model.predict(current_input)[0][0]
    future_prediations.append(future_pred)
    current_input = np.array([[future_pred]]).reshape(1, 1, 1)

print(f"Predicted next 3 numbers after 70: {future_prediations}")