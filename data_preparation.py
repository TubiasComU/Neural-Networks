import numpy as np
from sklearn.preprocessing import MinMaxScaler
from exchange_rates_loader import load_exchange_data

def prepare_data(sequence_length=10):
    df = load_exchange_data()

    # Save the original dates for later use
    dates = df["Date"].values

    # Normalize the data using MinMaxScaler
    scalers = {}
    scaled_data = {}
    for col in ["USD", "JPY"]:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
        scaled_data[col] = scaled

    # Create sequences for each currency
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    X_usd, y_usd = create_sequences(scaled_data["USD"], sequence_length)
    X_jpy, y_jpy = create_sequences(scaled_data["JPY"], sequence_length)

    # Divide the data into training and testing sets
    # 80% for training and 20% for testing
    split_usd = int(len(X_usd) * 0.8)
    split_jpy = int(len(X_jpy) * 0.8)

    data = {
        "usd": {
            "X_train": X_usd[:split_usd],
            "y_train": y_usd[:split_usd],
            "X_test": X_usd[split_usd:],
            "y_test": y_usd[split_usd:],
            "scaler": scalers["USD"],
        },
        "jpy": {
            "X_train": X_jpy[:split_jpy],
            "y_train": y_jpy[:split_jpy],
            "X_test": X_jpy[split_jpy:],
            "y_test": y_jpy[split_jpy:],
            "scaler": scalers["JPY"],
        },
    }

    return data

# Test the function
if __name__ == "__main__":
    data = prepare_data()
    print("USD treino:", data["usd"]["X_train"].shape)
    print("JPY treino:", data["jpy"]["X_train"].shape)
