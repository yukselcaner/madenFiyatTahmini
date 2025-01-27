from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Veri Hazırlığı ---
commodities = {
    'Altın': 'GC=F',       # Altın
    'Gümüş': 'SI=F',     # Gümüş
    'Platinyum': 'PL=F'    # Platin
}

currency_pair = 'USDTRY=X'
start_date = '2016-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Verileri İndirme ve İşleme
data_frames = {}
for name, ticker in commodities.items():
    data_frames[name] = yf.download(ticker, start=start_date, end=end_date)
usd_try_data = yf.download(currency_pair, start=start_date, end=end_date)

# Veri işleme ve gram fiyatı hesaplama
processed_data = {}
for name, df in data_frames.items():
    # MultiIndex sütunları düzleştir
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    usd_try_data.columns = [col[0] if isinstance(col, tuple) else col for col in usd_try_data.columns]

    # Tarih sütununu eklemek için index'i sıfırla
    df = df.reset_index()
    usd_try = usd_try_data.reset_index()

    # Sadece gerekli sütunları ve eksik değerleri kaldır
    df = df[['Date', 'Close']].dropna()
    df.rename(columns={'Close': f'{name}_Close'}, inplace=True)

    usd_try = usd_try[['Date', 'Close']].dropna()
    usd_try.rename(columns={'Close': 'Close_USDTRY'}, inplace=True)

    # Tarihleri hizalama ve birleştirme
    merged = pd.merge(df, usd_try, on='Date', how='outer')

    # Tarihleri sıralama ve eksik değerleri doldurma
    merged['Date'] = pd.to_datetime(merged['Date'])
    merged = merged.sort_values('Date').set_index('Date').asfreq('D')
    merged = merged.fillna(method='ffill')  # Eksik değerleri önceki değerle doldur

    # Gram fiyatı hesaplama
    if f'{name}_Close' in merged.columns and 'Close_USDTRY' in merged.columns:
        merged['Gram_Price'] = (merged[f'{name}_Close'] / 31.1) * merged['Close_USDTRY']
        processed_data[name] = merged[['Gram_Price']]
    else:
        print(f"Uyarı: {name} için gerekli sütunlar bulunamadı.")

def generate_plot(selected_commodity, forecast_period):
    df = processed_data[selected_commodity].reset_index()

    # Tahmin parametrelerini belirle
    if forecast_period == "Haftalık":
        past_days = 90  # 3 ay geçmiş veri
        future_steps = 7  # 1 hafta tahmin
    elif forecast_period == "Aylık":
        past_days = 365  # 1 yıl geçmiş veri
        future_steps = 30  # 1 ay tahmin
    elif forecast_period == "Yıllık":
        past_days = 1095  # 3 yıl geçmiş veri
        future_steps = 365  # 1 yıl tahmin
    else:
        raise ValueError("Geçersiz tahmin periyodu.")

    # SARIMA Modeli
    model = SARIMAX(df['Gram_Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit(disp=False)

    # Gelecek Tahmini
    future_index = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)
    forecast_future = model_fit.forecast(steps=future_steps)

    # Grafik başlangıç noktasını belirle
    start_plot_date = datetime.today() - timedelta(days=past_days)
    end_plot_date = future_index[-1]

    # Gelecek Tahmin Grafiği
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Gram_Price'], label='Gerçek Değer', color='gold')
    plt.plot(future_index, forecast_future, label='Gelecek Tahmin', linestyle='--', color='green')
    plt.xlim(left=start_plot_date, right=end_plot_date)
    plt.title(f'{selected_commodity} - Gerçek ve Gelecek Tahmin Değerler')
    plt.xlabel('Tarih')
    plt.ylabel('Gram Fiyat (TRY)')
    plt.legend()
    plt.grid()

    # Grafiği kaydet ve base64 olarak döndür
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()

    return plot_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    selected_commodity = "Altın"  # Varsayılan değer
    forecast_period = "Haftalık"  # Varsayılan değer

    if request.method == 'POST':
        selected_commodity = request.form.get('commodity')
        forecast_period = request.form.get('forecast_period')

    plot_url = generate_plot(selected_commodity, forecast_period)
    return render_template('model.html', 
                           selected_commodity=selected_commodity, 
                           forecast_period=forecast_period, 
                           commodities=commodities, 
                           plot_url=plot_url)

@app.route('/whous')
def whous():
    return render_template('whous.html')

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
