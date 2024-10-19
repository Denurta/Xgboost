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
        /* Change background color and font */
        body {
            background-color: #F5F5F5;
            font-family: 'Arial', sans-serif;
        }

        /* Style the main title */
        .stApp header {
            background: #4682B4; /* Steel Blue */
            padding: 10px;
        }

        /* Style the sidebar */
        .sidebar-content {
            background-color: #4682B4; /* Steel Blue */
            color: white;
        }

        /* Style buttons */
        .stButton>button {
            background-color: #4682B4; /* Steel Blue */
            color: white;
            border-radius: 12px;
        }

        /* Style sliders */
        .stSlider label {
            color: #4682B4; /* Steel Blue */
        }

        /* Add spacing between sections */
        .stMarkdown {
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Styling table */
        .dataframe {
            border: 1px solid #4682B4; /* Steel Blue */
            border-radius: 5px;
        }

        /* Make images responsive */
        img {
            max-width: 100%;
            height: auto;
        }

        </style>
        """, unsafe_allow_html=True)

# Function to display the title with a logo in a visually appealing way
def display_title_with_logo():
    st.markdown("""
        <style>
        .title-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 30px;
        }

        .title-container img {
            max-width: 80px;
            margin-right: 20px;
        }

        .title-container h1 {
            font-size: 3rem;
            color: #4682B4;
            margin: 0;
        }

        .stApp header {
            background-color: #F5F5F5; /* Light background */
        }
        </style>
        <div class="title-container">
            <img src="https://kuliahdimana.id/public/magang/22954b52c576f28055eac57190504a20.jpg" alt="Logo">
            <h1>Prediksi Saham LQ45</h1>
        </div>
        """, unsafe_allow_html=True)

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
    add_custom_css()

    # Sidebar navigation
    page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi Saham", "Penjelasan"])

    if page == "Prediksi Saham":
        display_title_with_logo()
        show_prediction_page()
    elif page == "Penjelasan Metrik dan Threshold":
        display_title_with_logo()
        show_explanation_page()

# Function to show prediction page
def show_prediction_page():
    st.markdown("<h2 style='color:#4682B4;'>Prediksi Harga Saham LQ45</h2>", unsafe_allow_html=True)

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
        st.write(f"<b>Data Harga Penutupan {selected_stock}:</b>", unsafe_allow_html=True)
        st.write(stock_data)

        dynamic_lags = select_dynamic_lags(stock_data['Close'], max_lags=20, threshold=threshold)

        # Button to perform prediction
        if st.button('Lakukan Peramalan'):
            y_test, y_pred, future_preds = xgboost_forecast(stock_data, forecast_days, dynamic_lags)

            # Create Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Harga Aktual'))
            fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Harga Prediksi')))
            future_dates = pd.date_range(y_test.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Prediksi Masa Depan', line=dict(dash='dot', color='green')))
            fig.update_layout(title=f'Prediksi Harga Saham {selected_stock} dan Ramalan {forecast_days} Hari', xaxis_title='Tanggal', yaxis_title='Harga Penutupan')
            st.plotly_chart(fig)

            # Display future predictions
            future_df = pd.DataFrame({'Tanggal': future_dates, 'Ramalan Harga': future_preds})
            st.write(f'Ramalan Harga untuk {forecast_days} Hari Mendatang:')
            st.dataframe(future_df)

# Function to show explanation page
def show_explanation_page():
    st.markdown("<h2 style='color:#4682B4;'>Penjelasan Metrik dan Threshold</h2>", unsafe_allow_html=True)
    st.write("""
    ### Penjelasan Metrik:
    
    - **Mean Squared Error (MSE)**: MSE mengukur seberapa jauh nilai prediksi dari nilai sebenarnya. Nilai MSE yang lebih kecil berarti prediksi model lebih akurat.
    
    - **Root Mean Squared Error (RMSE)**: RMSE adalah akar kuadrat dari MSE. Ini mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan data asli.
    
    - **Mean Absolute Error (MAE)**: MAE menunjukkan rata-rata selisih absolut antara nilai prediksi dan nilai sebenarnya. Nilai MAE yang lebih kecil berarti prediksi lebih akurat.
    
    - **R² (R-squared)**: R² mengukur seberapa baik model menjelaskan variasi data. Nilainya berkisar dari 0 hingga 1. Semakin mendekati 1, semakin baik model menjelaskan data.
    
    - **Mean Absolute Percentage Error (MAPE)**: MAPE menunjukkan persentase kesalahan rata-rata dalam prediksi. Semakin kecil nilainya, semakin akurat prediksinya.
    
    ### Penjelasan Threshold:
    
    - **Threshold Autokorelasi**: Threshold digunakan untuk menentukan seberapa kuat hubungan antara nilai saat ini dengan nilai sebelumnya dalam data. Semakin tinggi threshold, semakin ketat model dalam memilih lag yang signifikan. Contoh: Jika threshold diatur ke 0.2, hanya lag dengan autokorelasi lebih dari 0.2 yang dianggap signifikan.
    """)

# Function to get stock data using Yahoo Finance
def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Function to perform XGBoost forecasting
def xgboost_forecast(stock_data, forecast_days, dynamic_lags):
    # Prepare data for forecasting (using XGBoost)
    X = []
    y = []
    close_prices = stock_data['Close'].values
    for i in range(max(dynamic_lags), len(close_prices)):
        X.append([close_prices[i - lag] for lag in dynamic_lags])
        y.append(close_prices[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/Test Split
    split_index = len(X) - forecast_days
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Create and train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    # Predict future values
    y_pred = model.predict(X_test)
    future_preds = []
    
    # Iteratively predict for the forecast period
    for _ in range(forecast_days):
        last_known_data = [close_prices[-lag] for lag in dynamic_lags]
        next_pred = model.predict(np.array([last_known_data]))[0]
        future_preds.append(next_pred)
        close_prices = np.append(close_prices, next_pred)
    
    return pd.Series(y_test, index=stock_data.index[-forecast_days:]), pd.Series(y_pred, index=stock_data.index[-forecast_days:]), future_preds

# Run the main function
if __name__ == "__main__":
    main()
