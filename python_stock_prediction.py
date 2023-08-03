import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def prepare_data(data):
    # Create a new column "Next Close" containing the next day's closing price
    data["Next Close"] = data["Close"].shift(-1)
    # Drop rows with NaN values (last row with NaN in "Next Close")
    data = data.dropna()
    return data

def train_test_linear_regression(data):
    # Prepare features (X) and target variable (y)
    X = data[["Open", "High", "Low", "Close", "Volume"]]
    y = data["Next Close"]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate model performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return model

if __name__ == "__main__":
    # List of relevant stock symbols
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "LAC", "PLL"]  # Add more stock symbols as needed

    start_date = "2020-01-01"
    end_date = "2023-08-05" #ADJUST DATES!

    for stock_symbol in stock_symbols:
        stock_data = get_stock_data(stock_symbol, start_date, end_date)
        processed_data = prepare_data(stock_data)

        print(f"\nStock Symbol: {stock_symbol}")
        trained_model = train_test_linear_regression(processed_data)

        # Example prediction for the last row in the dataset
        last_row = processed_data.iloc[-1][["Open", "High", "Low", "Close", "Volume"]].values.reshape(1, -1)
        predicted_price = trained_model.predict(last_row)
        print(f"Predicted Next Close Price: {predicted_price[0]}")
