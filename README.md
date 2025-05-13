# Neural-Networks

A simple neural networks project to predict exchange rates using two models: a traditional BackPropagation neural network and an LSTM model.

## 📈 Objective

Predict daily exchange rates of USD/EUR and JPY/EUR using neural networks and compare the performance of two different model architectures.

## 📊 Dataset

Historical exchange rate data for USD/EUR and JPY/EUR was collected from [Investing.com](https://www.investing.com/). The dataset includes:

- Date
- Exchange rate (Price)

Only the daily closing prices were used.

## 🧹 Preprocessing

- Dates converted to datetime format
- Normalization using Min-Max scaling
- Time series windowing (5-day input to predict next day)
- 80% training / 20% testing split
- Separate models trained for USD and JPY

## 🧠 Models

- **BackPropagation Neural Network**: implemented from scratch using NumPy
- **LSTM**: implemented using TensorFlow/Keras

## 📉 Evaluation

Models were evaluated using accuracy, confusion matrix, and classification reports.

## 📁 Structure

/data → Contains raw and preprocessed exchange rate files
/models → Contains code for each model (BP and LSTM)
/outputs → Contains results, logs, and generated Excel files

## 📎 Requirements

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow (for LSTM)

## ⚙️ How to Run

1. Place the exchange rate CSVs in the `data/` folder  
2. Run the scripts inside `models/`  
3. Outputs will be generated in the `outputs/` folder

## 📄 License

This project is for academic purposes only.
