import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from data_preparation import prepare_data


def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate(data, label="usd"):
    X_train = data[label]["X_train"]
    y_train = data[label]["y_train"]
    X_test = data[label]["X_test"]
    y_test = data[label]["y_test"]

    model = build_model(input_shape=X_train.shape[1:])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    loss, mae = model.evaluate(X_test, y_test)
    print(f"{label.upper()} - Test MAE: {mae:.4f}")

    return model, history

if __name__ == "__main__":
    data = prepare_data(sequence_length=10)

    print("Treinar modelo para USD")
    model_usd, hist_usd = train_and_evaluate(data, label="usd")

    print("\nTreinar modelo para JPY")
    model_jpy, hist_jpy = train_and_evaluate(data, label="jpy")
