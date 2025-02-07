#many to one LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

data = np.array([5,10,15,20,25,30,35,40,45,50])
X, y = [], []

#use past 4 data to predict the next value
for i in range(len(data) - 4):
    X.append(data[i:i+4])
    y.append(data[i+4])

X = np.array(X).reshape((len(X), 4, 1))
y = np.array(y)

#build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(4, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#train the model\
model.fit(X, y, epochs=500, verbose=0)

#predict the next value
input_data = np.array([[35,40,45,50]]).reshape((1, 4, 1))
prediction = model.predict(input_data)
print(f"Predicted next value: {prediction[0][0]}")