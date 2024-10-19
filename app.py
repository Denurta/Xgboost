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
        significant_lags = [1, 5, 10]  # Default lags if none are significant
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

# Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Main function of the application
def main():
    st.set_page_config(page_title="Prediksi Saham LQ45", layout="wide", initial_sidebar_state="expanded")
    st.title("Prediksi Saham LQ45")
    st.sidebar.title("Menu Utama")

    # Initialize date input with default values
    start_date = st.sidebar.date_input('Tanggal Awal', pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input('Tanggal Akhir', pd.to_datetime('today'))

    selected_stock = st.sidebar.selectbox('Pilih Saham:', ['Pilih Saham'] + lq45_tickers, index=0)

    # Slider for the number of forecast days
    forecast_days = st.sidebar.slider('Jumlah Hari untuk Ramalan:', 1, 30, 1, format="%d hari")

    # Slider for threshold for autocorrelation
    threshold = st.sidebar.slider('Threshold:', 0.0, 1.0, 0.2, step=0.01, format="%.2f")

    if selected_stock != 'Pilih Saham':
        stock_data = get_stock_data(selected_stock, start_date, end_date)

        if stock_data.empty:
            st.warning(f"Tidak ada data tersedia untuk saham {selected_stock} pada periode yang dipilih.")
        else:
            st.write(f"Data Harga Penutupan {selected_stock}:")
            st.write(stock_data)

            dynamic_lags = select_dynamic_lags(stock_data['Close'], max_lags=20, threshold=threshold)

            # Button to perform prediction
            if st.button('Lakukan Peramalan'):
                y_test, y_pred, future_preds = xgboost_forecast(stock_data, forecast_days, dynamic_lags)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Harga Aktual'))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Harga Prediksi', line=dict(dash='dash')))

                future_dates = pd.date_range(y_test.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
                fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Prediksi Masa Depan', line=dict(dash='dot', color='green')))

                fig.update_layout(title=f'Prediksi Harga Saham {selected_stock} dan Ramalan {forecast_days} Hari',
                                  xaxis_title='Tanggal', yaxis_title='Harga Penutupan')
                st.plotly_chart(fig)

                future_df = pd.DataFrame({'Tanggal': future_dates, 'Ramalan Harga': future_preds})
                st.write(f'Ramalan Harga untuk {forecast_days} Hari Mendatang:')
                st.dataframe(future_df)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = calculate_mape(y_test, y_pred)

                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"R² Score: {r2:.2f}")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

                # Add explanation of metrics and threshold
                st.markdown("""
                ### Penjelasan Metrik:
                - **Mean Squared Error (MSE)**: Mengukur seberapa jauh prediksi dari nilai sebenarnya. Semakin kecil, semakin akurat.
    
                - **Root Mean Squared Error (RMSE)**: Akar kuadrat dari MSE. Menunjukkan besar kesalahan rata-rata. Semakin kecil, semakin baik.
    
                - **Mean Absolute Error (MAE)**: Rata-rata kesalahan absolut antara prediksi dan nilai sebenarnya. Semakin kecil, semakin akurat.
    
                - **R² (R-squared)**: Menunjukkan seberapa baik model menjelaskan data. Nilai 1 berarti sangat cocok, 0 berarti tidak cocok.
    
                - **Mean Absolute Percentage Error (MAPE)**: Mengukur kesalahan prediksi dalam persentase. Semakin kecil MAPE, semakin baik.

                ### Penjelasan Threshold:
                **Threshold** adalah nilai batas yang digunakan untuk menentukan apakah korelasi lag tertentu dianggap signifikan atau tidak. 
                Nilai korelasi yang lebih besar dari threshold dianggap penting dan digunakan untuk membuat model prediksi. 
                Anda dapat mengatur nilai threshold ini untuk menyesuaikan kepekaan model terhadap perubahan dalam data historis.
                """)
                
    else:
        st.write("Pilih saham untuk dianalisis.")

# Run the application
if __name__ == "__main__":
    main()
