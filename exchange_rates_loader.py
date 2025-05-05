import pandas as pd
import requests
from io import BytesIO
import zipfile


def load_exchange_data():
    # Download the ZIP file from ECB
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
    response = requests.get(url)
    z = zipfile.ZipFile(BytesIO(response.content))
    csv_file = z.open("eurofxref-hist.csv")

    # Read CSV
    df = pd.read_csv(csv_file)

    # Convert date column to datetime and sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Select relevant columns and rename them
    df = df[["Date", "USD", "JPY"]].dropna()

    print(df.tail())
    return df
