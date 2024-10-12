import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf

# Function to select dynamic lags based on autocorrelation
def select_dynamic_lags(price_data, max_lags=20, threshold=0.2):
    # Calculate autocorrelation for up to max_lags
    autocorr_values = acf(price_data, nlags=max_lags)
    
    # Select lags where autocorrelation is above a threshold (ignoring lag 0)
    significant_lags = [lag for lag, value in enumerate(autocorr_values[1:], start=1) if abs(value) > threshold]
    
    # If no significant lags, choose default lags (e.g., 1, 5, 10)
    if not significant_lags:
        significant_lags = [1, 5, 10]
    
    return significant_lags

# List of LQ45 stock tickers on Yahoo Finance
lq45_tickers = [
    'ACES.JK', 'ADRO.JK', 'AKRA.JK', 'AMMN.JK', 'AMRT.JK', 'ANTM.JK', 'ARTO.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK',
    'BBRI.JK', 'BBTN.JK', 'BMRI.JK', 'BRIS.JK', 'BRPT.JK', 'BUKA.JK', 'CPIN.JK', 'ESSA.JK', 'EXCL.JK', 'GGRM.JK',
    'GOTO.JK', 'HRUM.JK', 'ICBP.JK', 'INCO.JK', 'INDF.JK', 'INKP.JK', 'INTP.JK', 'ISAT.JK', 'ITMG.JK', 'JSMR.JK',
    'KLBF.JK', 'MAPI.JK', 'MBMA.JK', 'MDKA.JK', 'MEDC.JK', 'MTEL.JK', 'PGAS.JK', 'PGEO.JK', 'PTBA.JK', 'SIDO.JK',
    'SMGR.JK', 'TLKM.JK', 'TOWR.JK', 'UNTR.JK', 'UNVR.JK'
]

# Function to fetch stock data from Yahoo Finance
@st.cache_data
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    return data[['Close']]

# Function for forecasting using XGBoost
def xgboost_forecast(data, forecast_days, dynamic_lags):
    price_data = data['Close']
    
    # Determine lags dynamically
    lags = dynamic_lags if dynamic_lags else [1, 2, 3]
    
    lagged_data = {f'Lag_{lag}': price_data.shift(lag) for lag in lags}
    lagged_data_df = pd.DataFrame(lagged_data).dropna()

    X = lagged_data_df.values
    y = price_data[lagged_data_df.index]

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.1, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Predict future values
    last_known_data = X_test[-1].reshape(1, -1)
    future_preds = []

    for _ in range(forecast_days):
        next_pred = model.predict(last_known_data)[0]
        future_preds.append(next_pred)
        new_data = np.roll(last_known_data, -1)
        new_data[0, -1] = next_pred
        last_known_data = new_data

    return y_test, y_pred, future_preds

# Main function of the application
def main():
    # Set the page configuration
    st.set_page_config(page_title="Prediksi Saham LQ45", layout="wide", initial_sidebar_state="expanded")

    # Title of the application
    st.title("Prediksi Saham LQ45")
    st.sidebar.title("Main Menu")

    # Select stock to analyze
    selected_stock = st.sidebar.selectbox('Pilih saham:', lq45_tickers)

    # Select start and end dates
    start_date = st.sidebar.date_input('Tanggal mulai', pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input('Tanggal akhir', pd.to_datetime('today'))

    # Select the number of days for forecasting
    forecast_days = st.sidebar.slider('Jumlah hari untuk ramalan:', 1, 30, 7)

    # Input threshold for lag selection
    threshold = st.sidebar.slider('Threshold untuk autocorrelation:', 0.0, 1.0, 0.2, 0.01)

    # Fetch stock data
    stock_data = get_stock_data(selected_stock, start_date, end_date)

    # Display stock data
    st.write(f"Data harga penutupan {selected_stock}:")
    st.write(stock_data)

    # Select lags dynamically
    dynamic_lags = select_dynamic_lags(stock_data['Close'], max_lags=20, threshold=threshold)

    # Forecast with XGBoost
    if st.button('Lakukan Peramalan'):
        y_test, y_pred, future_preds = xgboost_forecast(stock_data, forecast_days, dynamic_lags)

        # Plot results (Actual vs Predicted)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual Prices'))
        fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted Prices', line=dict(dash='dash')))

        # Plot future predictions
        future_dates = pd.date_range(y_test.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Future Predictions', line=dict(dash='dot', color='green')))

        fig.update_layout(title=f'Prediksi Harga Saham {selected_stock} dan Ramalan {forecast_days} Hari',
                          xaxis_title='Tanggal', yaxis_title='Harga Penutupan')
        st.plotly_chart(fig)

        # Display future predictions in a table
        future_df = pd.DataFrame({'Tanggal': future_dates, 'Ramalan Harga': future_preds})
        st.write(f'Ramalan Harga untuk {forecast_days} Hari Mendatang:')
        st.dataframe(future_df)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display evaluation metrics
        st.write(f"RMSE: {np.sqrt(mse):.2f}")
        st.write(f"RMAE: {mae:.2f}")
        st.write(f"R²: {r2:.2f}")

# Run the application
if __name__ == "__main__":
    main()
