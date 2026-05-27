# Modul 11 · Time Series Forecasting

> Memprediksi masa depan dari pola masa lalu: penjualan, harga, permintaan, trafik, sensor IoT. Time series **berbeda fundamental** dari ML biasa karena **urutan & waktu itu penting** — kamu tidak boleh shuffle data, dan banyak intuisi ML standar justru menyesatkan di sini.

## Tujuan Belajar
- Memahami komponen time series & kenapa ia "spesial".
- Melakukan **feature engineering temporal** (lag, rolling, kalender).
- Menggunakan **validasi yang benar** (tidak boleh acak!).
- Membandingkan pendekatan: statistik klasik (ARIMA), ML (XGBoost), dan modern (Prophet/deep learning).
- Menghindari jebakan klasik: **data leakage waktu**.

## Daftar Isi
1. Apa yang membuat time series berbeda
2. Komponen: tren, musiman, noise
3. Konsep penting: stationarity & autokorelasi
4. Feature engineering temporal ⭐
5. Validasi time series (JANGAN shuffle!)
6. Pendekatan 1 — Statistik klasik (ARIMA)
7. Pendekatan 2 — ML (XGBoost dengan fitur lag)
8. Pendekatan 3 — Prophet & deep learning
9. Metrik evaluasi forecasting
10. Jebakan & cheat sheet

---

## 1. Apa yang Membuat Time Series Berbeda

| ML biasa | Time series |
|---|---|
| Sampel dianggap independen | Sampel **bergantung** pada urutan waktu |
| Boleh di-shuffle | **Tidak boleh** shuffle — urutan = informasi |
| Split acak | Split **kronologis** (latih masa lalu, uji masa depan) |
| Tidak ada "bocor dari masa depan" | Leakage waktu adalah bahaya utama |

> Aturan emas: **kamu hanya boleh memakai informasi yang benar-benar tersedia pada saat prediksi.** Memakai data masa depan (sengaja/tak sengaja) = leakage yang membuat model hebat di backtest tapi gagal nyata.

---

## 2. Komponen Time Series

Sebuah deret biasanya bisa diuraikan jadi:
- **Tren (trend)** — arah jangka panjang (naik/turun).
- **Musiman (seasonality)** — pola berulang dengan periode tetap (harian, mingguan, tahunan).
- **Siklus (cyclic)** — pola berulang tanpa periode tetap (mis. siklus ekonomi).
- **Residual/noise** — sisa yang tak terjelaskan.

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("penjualan.csv", parse_dates=["tanggal"], index_col="tanggal")
result = seasonal_decompose(df["penjualan"], model="additive", period=12)
result.plot()   # lihat trend, seasonal, residual terpisah
```

**Additive vs multiplicative:** kalau amplitudo musiman **konstan** → additive; kalau **membesar seiring tren** → multiplicative (atau transform `log` dulu).

---

## 3. Konsep Penting: Stationarity & Autokorelasi

### Stationarity
Banyak model klasik (ARIMA) mengasumsikan deret **stasioner** (mean & variansi stabil sepanjang waktu). Tren/musiman membuatnya tidak stasioner → perlu di-**differencing** (`y_t − y_{t-1}`).

```python
from statsmodels.tsa.stattools import adfuller
p_value = adfuller(df["penjualan"])[1]
# p < 0.05 -> stasioner; kalau tidak, lakukan differencing:
df["diff"] = df["penjualan"].diff()
```

### Autokorelasi (ACF/PACF)
Mengukur korelasi deret dengan dirinya sendiri di lag berbeda — membantu menemukan periode musiman & memilih parameter model.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df["penjualan"]); plot_pacf(df["penjualan"])
# puncak di lag 7 -> pola mingguan; lag 12 -> pola tahunan (data bulanan)
```

---

## 4. Feature Engineering Temporal ⭐ (kunci pendekatan ML)

Untuk memakai model ML (XGBoost dll), kita ubah time series jadi tabel fitur — **ini sering pendekatan paling kuat & praktis**.

```python
df = df.sort_index()   # PASTIKAN urut waktu!

# a) Fitur kalender (dari timestamp)
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
df["is_month_start"] = df.index.is_month_start.astype(int)

# b) Fitur LAG — nilai masa lalu (paling penting!)
for lag in [1, 7, 14, 30]:
    df[f"lag_{lag}"] = df["penjualan"].shift(lag)

# c) Rolling window — statistik bergerak (HARUS pakai shift agar tak bocor!)
df["roll_mean_7"] = df["penjualan"].shift(1).rolling(7).mean()
df["roll_std_7"]  = df["penjualan"].shift(1).rolling(7).std()
df["roll_max_30"] = df["penjualan"].shift(1).rolling(30).max()

# d) Fitur eksternal (jika ada): cuaca, hari libur, promo, harga
df = df.dropna()   # baris awal punya NaN karena lag/rolling
```

> 🚨 **Perhatikan `.shift(1)` sebelum `.rolling()`.** Tanpa itu, rolling mean menyertakan nilai hari ini → kamu memakai jawaban untuk memprediksi jawaban = **leakage**. Ini kesalahan paling umum di forecasting.

---

## 5. Validasi Time Series (JANGAN pakai shuffle / k-fold biasa!)

Split acak **bocor masa depan ke masa lalu**. Gunakan split **kronologis** atau **TimeSeriesSplit** (expanding window).

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # train selalu SEBELUM test secara waktu
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    ...
```

```
Fold 1: [train][test].................
Fold 2: [train train][test]...........   <- jendela membesar, test selalu di masa depan
Fold 3: [train train train][test].....
        ───────────────── waktu →
```

Untuk evaluasi akhir, **backtesting**: latih sampai titik T, prediksi T+1..T+h, geser maju, ulangi.

---

## 6. Pendekatan 1 — Statistik Klasik (ARIMA / SARIMA)

Model time series tradisional. Bagus untuk deret tunggal dengan pola jelas & data tidak terlalu banyak.

- **ARIMA(p, d, q)**: AR (autoregresif) + I (differencing) + MA (moving average).
- **SARIMA**: ARIMA + komponen musiman.

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train["penjualan"], order=(2, 1, 2))   # (p, d, q)
fitted = model.fit()
forecast = fitted.forecast(steps=30)

# auto-pilih parameter: pip install pmdarima
import pmdarima as pm
auto = pm.auto_arima(train["penjualan"], seasonal=True, m=12)
```

✅ Interpretable, baik untuk data kecil. ❌ Sulit memakai fitur eksternal, satu deret per model.

---

## 7. Pendekatan 2 — ML dengan Fitur Lag (XGBoost) ⭐

Sering **pemenang praktis** untuk forecasting bisnis: ubah jadi regresi tabular (bagian 4), lalu pakai model kuat. Mudah menambah fitur eksternal & menangani banyak deret.

```python
from xgboost import XGBRegressor

features = [c for c in df.columns if c != "penjualan"]
X, y = df[features], df["penjualan"]

# Split kronologis (BUKAN acak)
split = int(len(df) * 0.8)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)
```

⚠️ **Multi-step forecast:** untuk prediksi jauh ke depan, kamu tak punya lag aktual masa depan → pakai strategi *recursive* (umpan balik prediksi sebagai lag berikutnya) atau *direct* (model terpisah per horizon).

✅ Akurat, fleksibel, multi-fitur. ❌ Butuh feature engineering hati-hati, tak otomatis tangani tren ekstrapolasi.

---

## 8. Pendekatan 3 — Prophet & Deep Learning

### Prophet (Meta)
Mudah dipakai, otomatis menangani tren + musiman + hari libur. Bagus sebagai baseline cepat untuk data bisnis.

```python
# pip install prophet
from prophet import Prophet

dfp = df.reset_index().rename(columns={"tanggal": "ds", "penjualan": "y"})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.add_country_holidays(country_name="ID")
m.fit(dfp)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)        # termasuk interval ketidakpastian
m.plot(forecast); m.plot_components(forecast)
```

### Deep Learning
Untuk pola sangat kompleks, banyak deret, atau data berlimpah:
- **LSTM/GRU** (RNN) — klasik untuk sequence.
- **Temporal Fusion Transformer, N-BEATS, N-HiTS** — state-of-the-art modern.
- Library: **Darts**, **GluonTS**, **PyTorch Forecasting** (mengemas banyak model + validasi time series dengan benar).

```python
# Contoh konsep dengan Darts (pip install darts) — API tinggi-level:
# from darts import TimeSeries
# from darts.models import NBEATSModel
# series = TimeSeries.from_dataframe(df, value_cols="penjualan")
# model = NBEATSModel(input_chunk_length=30, output_chunk_length=7).fit(series)
```

---

## 9. Metrik Evaluasi Forecasting

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_te, pred)
rmse = np.sqrt(mean_squared_error(y_te, pred))
mape = np.mean(np.abs((y_te - pred) / y_te)) * 100   # % error, mudah dipahami bisnis
```

| Metrik | Catatan |
|---|---|
| **MAE** | error rata-rata, satuan sama dengan target |
| **RMSE** | menghukum error besar |
| **MAPE** | error persentase — enak untuk laporan bisnis, tapi pecah jika ada nilai 0 |
| **sMAPE / WAPE** | varian MAPE yang lebih stabil |

> Selalu bandingkan dengan **baseline naif**: "prediksi = nilai kemarin" atau "= rata-rata musiman". Kalau modelmu tak mengalahkan ini, ia belum berguna.

---

## 10. Jebakan & Cheat Sheet

**Jebakan paling umum:**
1. **Shuffle data / k-fold acak** → bocor masa depan. Pakai TimeSeriesSplit.
2. **Rolling tanpa shift** → memakai nilai hari ini. Selalu `.shift(1)` dulu.
3. **Scaling/fitting pakai seluruh data** → statistik masa depan bocor. Fit di train saja.
4. **Lupa baseline naif** → tak tahu apakah model benar-benar menambah nilai.
5. **Mengabaikan ketidakpastian** → laporkan interval, bukan satu angka.

**Pilih pendekatan:**
```
Data kecil, satu deret, pola jelas      → ARIMA/SARIMA atau Prophet
Banyak fitur eksternal / banyak deret    → XGBoost dengan fitur lag ⭐
Baseline bisnis cepat (tren+musiman+libur) → Prophet
Pola sangat kompleks, data berlimpah     → Deep learning (Darts/PyTorch Forecasting)
```

## Latihan
1. Ambil dataset time series (mis. Air Passengers, penjualan retail, atau harga saham). Dekomposisi jadi tren/musiman/residual.
2. Buat fitur lag + rolling (dengan `.shift` yang benar!) dan latih XGBoost. Bandingkan dengan baseline naif "nilai kemarin".
3. Tunjukkan bahaya leakage: bandingkan validasi dengan k-fold acak vs TimeSeriesSplit. Kenapa yang acak menipu?
4. Bandingkan ARIMA, Prophet, dan XGBoost pada deret yang sama. Buat tabel MAE/RMSE/MAPE.
5. Lakukan multi-step forecast 30 hari ke depan dengan strategi recursive. Plot prediksi + interval.

⬅️ Kembali ke [Daftar Modul](../README.md) · Terkait: [Modul 02 · Feature Engineering](../02-data-feature-engineering/README.md), [Modul 03 · Classical ML](../03-classical-ml/README.md)
