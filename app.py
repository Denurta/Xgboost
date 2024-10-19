import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import acf
import yfinance as yf
import plotly.graph_objects as go

# Set page configuration here, before the main function
st.set_page_config(page_title="Prediksi Saham LQ45", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve styling
def add_custom_css():
    st.markdown("""
        <style>
        body {
            background-color: #F5F5F5;
            font-family: 'Arial', sans-serif;
        }
        .stApp header {
            background: #4682B4; 
            padding: 10px;
        }
        .sidebar-content {
            background-color: #4682B4;
            color: white;
        }
        .stButton>button {
            background-color: #4682B4;
            color: white;
            border-radius: 12px;
        }
        .stSlider label, .stSelectbox label, .stDateInput label {
            color: #4682B4;
        }
        .stSidebar .stSelectbox div, .stSidebar .stDateInput div, .stSidebar .stSlider div {
            color: #4682B4;
        }
        .stMarkdown {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .dataframe {
            border: 1px solid #4682B4;
            border-radius: 5px;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        </style>
        """, unsafe_allow_html=True)

# Function to select dynamic lags based on autocorrelation
def select_dynamic_lags(price_data, max_lags=20, threshold=0.2):
    autocorr_values = acf(price_data, nlags=max_lags)
    significant_lags = [lag for lag, value in enumerate(autocorr_values[1:], start=1) if abs(value) > threshold]
    return significant_lags if significant_lags else [1, 5, 10]

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
    add_custom_css()

    # Sidebar navigation
    page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi Saham", "Penjelasan"])

    if page == "Prediksi Saham":
        show_prediction_page()
    elif page == "Penjelasan":
        show_explanation_page()

# Function to show prediction page
def show_prediction_page():
    st.markdown("<h2 style='color:#4682B4;'>Prediksi Harga Saham LQ45</h2>", unsafe_allow_html=True)

    # Initialize date input with default values
    start_date = st.sidebar.date_input('Tanggal Awal', pd.to_datetime('2019-01-01'))
    end_date = st.sidebar.date_input('Tanggal Akhir', pd.to_datetime('today'))

    selected_stock = st.sidebar.selectbox('Pilih Saham:', ['Pilih Saham'] + lq45_tickers, index=0)

    # Slider for threshold for autocorrelation
    threshold = st.sidebar.slider('Threshold Autokorelasi:', 0.0, 1.0, 0.2, step=0.01, format="%.2f")
    
    # Slider for the number of forecast days
    forecast_days = st.sidebar.slider('Jumlah Hari untuk Ramalan:', 1, 30, 1, format="%d hari")

    if selected_stock != 'Pilih Saham':
        stock_data = get_stock_data(selected_stock, start_date, end_date)
        st.write(f"<b>Data Harga Penutupan {selected_stock}:</b>", unsafe_allow_html=True)
        st.dataframe(stock_data)

        dynamic_lags = select_dynamic_lags(stock_data['Close'], max_lags=20, threshold=threshold)

        # Button to perform prediction
        if st.button('Lakukan Peramalan'):
            y_test, y_pred, future_preds = xgboost_forecast(stock_data, forecast_days, dynamic_lags)

            # Create Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Harga Aktual'))
            fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Harga Prediksi', line=dict(dash='dot')))
            future_dates = pd.date_range(y_test.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Prediksi Masa Depan', line=dict(dash='dot', color='red')))
            fig.update_layout(title=f'Prediksi Harga Saham {selected_stock} dan Ramalan {forecast_days} Hari', xaxis_title='Tanggal', yaxis_title='Harga Penutupan')
            st.plotly_chart(fig)

            # Display future predictions
            future_df = pd.DataFrame({'Tanggal': future_dates, 'Ramalan Harga': future_preds})
            st.write(f'Ramalan Harga untuk {forecast_days} Hari Mendatang:')
            st.dataframe(future_df)

# Function to show explanation page
def show_explanation_page():
    st.markdown("<h2 style='color:#4682B4;'>Penjelasan</h2>", unsafe_allow_html=True)
    st.write("""
     ### Penjelasan Threshold:
    
    - **Threshold Autokorelasi**: Threshold digunakan untuk menentukan seberapa kuat hubungan antara nilai saat ini dengan nilai sebelumnya dalam data. Semakin tinggi threshold, semakin ketat model dalam memilih lag yang signifikan. Contoh: Jika threshold diatur ke 0.2, hanya lag dengan autokorelasi lebih dari 0.2 yang akan digunakan.
    
    ### Penjelasan Metrik:
    
    - **Mean Squared Error (MSE)**: MSE mengukur seberapa jauh nilai prediksi dari nilai sebenarnya. Nilai MSE yang lebih kecil berarti prediksi model lebih akurat.
    
    - **Root Mean Squared Error (RMSE)**: RMSE adalah akar kuadrat dari MSE. Ini mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan data asli.
    
    - **Mean Absolute Error (MAE)**: MAE menunjukkan rata-rata selisih absolut antara nilai prediksi dan nilai sebenarnya. Nilai MAE yang lebih kecil berarti prediksi lebih akurat.
    
    - **R² (R-squared)**: R² mengukur seberapa baik model menjelaskan variasi data. Nilainya berkisar dari 0 hingga 1. Semakin mendekati 1, semakin baik model menjelaskan data.
    
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

    model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    last_known_values = X_test[-1].reshape(1, -1)
    future_preds = []
    for i in range(forecast_days):
        next_pred = model.predict(last_known_values)[0]
        future_preds.append(next_pred)
        last_known_values = np.roll(last_known_values, -1)
        last_known_values[0, -1] = next_pred

    return y_test, y_pred, future_preds

if __name__ == '__main__':
    main()
