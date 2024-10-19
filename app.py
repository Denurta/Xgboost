import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import acf
import yfinance as yf
import plotly.graph_objects as go

# Function to select dynamic lags based on autocorrelation
def select_dynamic_lags(price_data, max_lags=20, threshold=0.2):
    autocorr_values = acf(price_data, nlags=max_lags)
    significant_lags = [lag for lag, value in enumerate(autocorr_values[1:], start=1) if abs(value) > threshold]
    if not significant_lags:
        significant_lags = [1, 5, 10]
    return significant_lags

# List of LQ45 stock tickers on Yahoo Finance
lq45_tickers = [
    'ACES.JK', 'ADRO.JK', 'AKRA.JK', 'AMMN.JK', 'AMRT.JK', 'ANTM.JK',
    'ARTO.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK', 'BBTN.JK',
    'BMRI.JK', 'BRIS.JK', 'BRPT.JK', 'BUKA.JK', 'CPIN.JK', 'ESSA.JK',
    'EXCL.JK', 'GGRM.JK', 'GOTO.JK', 'HRUM.JK', 'ICBP.JK', 'INCO.JK',
    'INDF.JK', 'INKP.JK', 'INTP.JK', 'ISAT.JK', 'ITMG.JK', 'JSMR.JK',
    'KLBF.JK', 'MAPI.JK', 'MBMA.JK', 'MDKA.JK', 'MEDC.JK', 'MTEL.JK',
    'PGAS.JK', 'PGEO.JK', 'PTBA.JK', 'SIDO.JK', 'SMGR.JK', 'TLKM.JK',
    'TOWR.JK', 'UNTR.JK', 'UNVR.JK'
]

# Main function for multi-page navigation
def main():
    st.set_page_config(page_title="Prediksi Saham LQ45", layout="wide", initial_sidebar_state="expanded")

    # Sidebar navigation
    page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi Saham", "Penjelasan Metrik dan Threshold"])

    if page == "Prediksi Saham":
        show_prediction_page()
    elif page == "Penjelasan Metrik dan Threshold":
        show_explanation_page()

# Function to show prediction page
def show_prediction_page():
    st.title("Prediksi Saham LQ45")
    st.sidebar.title("Menu Utama")

    # Initialize date input with default values
    start_date = st.sidebar.date_input('Tanggal Awal', pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input('Tanggal Akhir', pd.to_datetime('today'))

    selected_stock = st.sidebar.selectbox('Pilih Saham:', ['Pilih Saham'] + lq45_tickers, index=0)

    # Slider for the number of forecast days
    forecast_days = st.sidebar.slider('Jumlah Hari untuk Ramalan:', 1, 30, 1, format="%d hari")

    # Slider for threshold for autocorrelation
    threshold = st.sidebar.slider('Threshold:', 0.0, 1.0, 0.00, step=0.01, format="%.2f")

    if selected_stock != 'Pilih Saham':
        stock_data = get_stock_data(selected_stock, start_date, end_date)
        st.write(f"Data Harga Penutupan {selected_stock}:")
        st.write(stock_data)

        dynamic_lags = select_dynamic_lags(stock_data['Close'], max_lags=20, threshold=threshold)

        # Button to perform prediction
        if st.button('Lakukan Peramalan'):
            y_test, y_pred, future_preds = xgboost_forecast(stock_data, forecast_days, dynamic_lags)
            # Plotting and displaying the forecast data...

# Function to show explanation page
def show_explanation_page():
    st.title("Penjelasan Metrik dan Threshold")

    st.write("""
    ### Penjelasan Metrik:
    
    - **Mean Squared Error (MSE)**: MSE mengukur seberapa jauh nilai prediksi dari nilai sebenarnya. Nilai MSE yang lebih kecil berarti prediksi model lebih akurat.
    
    - **Root Mean Squared Error (RMSE)**: RMSE adalah akar kuadrat dari MSE. Ini mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan data asli.
    
    - **Mean Absolute Error (MAE)**: MAE menunjukkan rata-rata selisih absolut antara nilai prediksi dan nilai sebenarnya. Nilai MAE yang lebih kecil berarti prediksi lebih akurat.
    
    - **R² (R-squared)**: R² mengukur seberapa baik model menjelaskan variasi data. Nilainya berkisar dari 0 hingga 1. Semakin mendekati 1, semakin baik model menjelaskan data.
    
    - **Mean Absolute Percentage Error (MAPE)**: MAPE menunjukkan persentase kesalahan rata-rata dalam prediksi. Semakin kecil nilainya, semakin akurat prediksinya.
    
    ### Penjelasan Threshold:
    
    - **Threshold Autokorelasi**: Threshold digunakan untuk menentukan seberapa kuat hubungan antara nilai saat ini dengan nilai sebelumnya dalam data. Semakin tinggi threshold, semakin ketat model dalam memilih lag yang signifikan. Contoh: Jika threshold diatur ke 0.2, hanya lag dengan autokorelasi lebih dari 0.2 yang akan digunakan.
    """)

# Function to fetch stock data from Yahoo Finance with caching
@st.cache_data
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    return data[['Close']]

# Function for forecasting using XGBoost
def xgboost_forecast(data, forecast_days, dynamic_lags):
    price_data = data['Close']
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
    future_preds = []
    last_known_data = X_test[-1].reshape(1, -1)

    for _ in range(forecast_days):
        next_pred = model.predict(last_known_data)[0]
        future_preds.append(next_pred)
        new_data = np.roll(last_known_data, -1)
        new_data[0, -1] = next_pred
        last_known_data = new_data

    return y_test, y_pred, future_preds

# Run the application
if __name__ == "__main__":
    main()
