# Understanding Sequential Data
import numpy as np

#creatinf a simple sequence dataset
data = np.array([10, 20, 30, 40, 50, 60, 70])
X = []
y = []
#creatinf input-output pairs for sequence learning
for i in range(len(data) - 1):
    X.append(data[i])
    y.append(data[i + 1])

#reshaping the data to (samples, time_steps, features)
X = np.array(X).reshape((len(X), 1, 1))
y = np.array(y)

print("X (input of LSTM):", X)
print("y (Expected output):", y)