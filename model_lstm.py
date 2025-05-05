import numpy as np
import matplotlib.pyplot as plt
from data_preparation import prepare_data
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare data
data = prepare_data(sequence_length=10)

# USD and JPY data
X_train_usd = data["usd"]["X_train"]
y_train_usd = data["usd"]["y_train"]
X_test_usd = data["usd"]["X_test"]
y_test_usd = data["usd"]["y_test"]

X_train_jpy = data["jpy"]["X_train"]
y_train_jpy = data["jpy"]["y_train"]
X_test_jpy = data["jpy"]["X_test"]
y_test_jpy = data["jpy"]["y_test"]

# Reshape the data for LSTM
X_train_usd = X_train_usd.reshape((X_train_usd.shape[0], X_train_usd.shape[1], 1))
X_test_usd = X_test_usd.reshape((X_test_usd.shape[0], X_test_usd.shape[1], 1))

X_train_jpy = X_train_jpy.reshape((X_train_jpy.shape[0], X_train_jpy.shape[1], 1))
X_test_jpy = X_test_jpy.reshape((X_test_jpy.shape[0], X_test_jpy.shape[1], 1))

# Combine USD and JPY data for both training and testing
X_train = np.concatenate((X_train_usd, X_train_jpy), axis=1)
y_train = np.concatenate((y_train_usd, y_train_jpy), axis=1)

X_test = np.concatenate((X_test_usd, X_test_jpy), axis=1)
y_test = np.concatenate((y_test_usd, y_test_jpy), axis=1)

# Model LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(2))  # Output layer for 2 currencies (USD and JPY)

model.compile(optimizer='adam', loss='mse')
model.summary()

# Model training
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Model evaluation
y_pred = model.predict(X_test)

# Separate the predictions for USD and JPY
y_pred_usd = y_pred[:, 0]
y_pred_jpy = y_pred[:, 1]

# Calculate MSE for USD and JPY
mse_usd = mean_squared_error(y_test_usd, y_pred_usd)
print(f"Test MSE for USD: {mse_usd:.4f}")

mse_jpy = mean_squared_error(y_test_jpy, y_pred_jpy)
print(f"Test MSE for JPY: {mse_jpy:.4f}")

# Graphical representation of the results for USD
plt.figure(figsize=(10, 5))
plt.plot(y_test_usd, label="USD Real", marker='o')
plt.plot(y_pred_usd, label="USD Previsto", marker='x')
plt.title("Previsão LSTM - USD")
plt.xlabel("Índice")
plt.ylabel("Taxa de Câmbio (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# Graphical representation of the results for JPY
plt.figure(figsize=(10, 5))
plt.plot(y_test_jpy, label="JPY Real", marker='o')
plt.plot(y_pred_jpy, label="JPY Previsto", marker='x')
plt.title("Previsão LSTM - JPY")
plt.xlabel("Índice")
plt.ylabel("Taxa de Câmbio (JPY)")
plt.legend()
plt.tight_layout()
plt.show()
