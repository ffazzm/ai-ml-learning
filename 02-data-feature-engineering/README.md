# Modul 02 · Data & Feature Engineering

> "Applied machine learning is basically feature engineering." — Andrew Ng. Model canggih dengan fitur jelek kalah dari model sederhana dengan fitur bagus. Modul ini soal **mengubah data mentah jadi sinyal yang bisa dipelajari model**.

## Tujuan Belajar
- Melakukan **EDA** (Exploratory Data Analysis) yang sistematis.
- Menangani **missing values, outlier, dan data kotor**.
- Melakukan **encoding** untuk variabel kategorikal.
- Melakukan **scaling/normalisasi** dengan benar (tanpa leakage).
- Membuat fitur baru & melakukan **feature selection**.
- Membungkus semuanya dalam **Pipeline** scikit-learn.

---

## 1. Exploratory Data Analysis (EDA)

Sebelum modeling, **pahami datamu**. EDA = mengajukan pertanyaan ke data.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

# Pemahaman struktur
df.shape                 # berapa baris & kolom?
df.info()                # tipe data, missing
df.describe()            # statistik numerik
df.describe(include="object")   # statistik kategorikal
df.nunique()             # kardinalitas tiap kolom

# Target & keseimbangan kelas
df["target"].value_counts(normalize=True)   # apakah seimbang?
```

### Visualisasi EDA yang wajib
```python
# Distribusi tiap fitur numerik
df.hist(bins=30, figsize=(15, 10)); plt.tight_layout()

# Korelasi antar fitur (cari fitur redundan & yang berkaitan dengan target)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0)

# Hubungan fitur vs target
sns.boxplot(data=df, x="target", y="fitur_numerik")
sns.countplot(data=df, x="fitur_kategori", hue="target")

# Deteksi outlier
sns.boxplot(data=df, y="harga")
```

### Pertanyaan EDA yang harus dijawab
- Berapa banyak missing per kolom? Polanya acak atau sistematis?
- Apakah ada outlier? Wajar atau error input?
- Apakah target seimbang?
- Fitur mana paling berkorelasi dengan target?
- Apakah ada fitur yang bocor (leakage) — terlalu "sempurna" memprediksi target?

---

## 2. Menangani Missing Values

```python
df.isnull().sum()                          # hitung missing
df.isnull().mean().sort_values(ascending=False)   # proporsi missing
```

**Strategi:**

| Situasi | Tindakan |
|---|---|
| Missing sedikit (<5%) | hapus baris (`dropna`) atau imputasi |
| Kolom hampir kosong (>60%) | pertimbangkan buang kolom |
| Numerik | isi dengan **median** (tahan outlier) atau mean |
| Kategorikal | isi dengan **modus** atau kategori "Unknown" |
| Missing bermakna | buat fitur indikator `is_missing` |

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
```

> ⚠️ Imputasi harus di-`fit` pada **data train saja**, lalu diterapkan ke val/test. Lihat bagian Pipeline.

---

## 3. Menangani Outlier

```python
# Metode IQR (Interquartile Range)
Q1 = df["harga"].quantile(0.25)
Q3 = df["harga"].quantile(0.75)
IQR = Q3 - Q1
batas_bawah = Q1 - 1.5 * IQR
batas_atas = Q3 + 1.5 * IQR

outliers = df[(df["harga"] < batas_bawah) | (df["harga"] > batas_atas)]
```

**Pilihan penanganan:** hapus (jika error), **capping/clipping** (batasi ke nilai maksimum wajar), transformasi (log), atau biarkan (jika model tahan outlier seperti tree-based).

```python
df["harga"] = df["harga"].clip(lower=batas_bawah, upper=batas_atas)   # capping
import numpy as np
df["harga_log"] = np.log1p(df["harga"])   # log1p = log(1+x), aman untuk 0
```

---

## 4. Encoding Variabel Kategorikal

Model butuh angka. Ubah kategori → numerik.

### One-Hot Encoding (kategori nominal, tanpa urutan)
```python
# Untuk: warna [merah, hijau, biru] -> 3 kolom biner
df_encoded = pd.get_dummies(df, columns=["warna"], drop_first=True)

# atau scikit-learn (lebih baik untuk pipeline):
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```
⚠️ Hati-hati kardinalitas tinggi (mis. kode pos) → ledakan kolom.

### Ordinal / Label Encoding (kategori berurutan)
```python
# Untuk: ukuran [kecil < sedang < besar] -> 0, 1, 2
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[["kecil", "sedang", "besar"]])
```

### Target Encoding (kardinalitas tinggi)
Ganti kategori dengan rata-rata target untuk kategori itu. Kuat tapi **rawan leakage** — wajib pakai cross-validation/smoothing.

| Teknik | Kapan |
|---|---|
| One-Hot | nominal, kardinalitas rendah |
| Ordinal | ada urutan natural |
| Target encoding | kardinalitas tinggi (hati-hati leakage) |

---

## 5. Feature Scaling

Banyak algoritma (regresi, SVM, KNN, neural net) sensitif terhadap **skala** fitur. Fitur "gaji" (jutaan) akan mendominasi "umur" (puluhan) kalau tidak diskalakan.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
```

| Scaler | Rumus | Pakai saat |
|---|---|---|
| **StandardScaler** | (x − μ) / σ | default, data ~normal |
| **MinMaxScaler** | (x − min)/(max − min) → [0,1] | butuh range tetap (mis. neural net) |
| **RobustScaler** | pakai median & IQR | banyak outlier |

> 🚨 **Aturan anti-leakage:** `fit` scaler **hanya** di train, lalu `transform` train, val, dan test dengan statistik train.
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)         # HANYA transform
```

> 📌 **Tree-based model (Decision Tree, Random Forest, XGBoost) TIDAK butuh scaling** — mereka berbasis pemisahan threshold, bukan jarak.

---

## 6. Feature Engineering (membuat fitur baru)

Di sinilah pengetahuan domain berharga. Contoh:

```python
# Fitur dari tanggal
df["tanggal"] = pd.to_datetime(df["tanggal"])
df["hari_dalam_minggu"] = df["tanggal"].dt.dayofweek
df["bulan"] = df["tanggal"].dt.month
df["akhir_pekan"] = (df["tanggal"].dt.dayofweek >= 5).astype(int)

# Fitur rasio & interaksi (sering sangat informatif)
df["harga_per_m2"] = df["harga"] / df["luas"]
df["rasio_kamar"] = df["kamar_tidur"] / df["kamar_mandi"]

# Binning (kontinu -> kategori)
df["kelompok_umur"] = pd.cut(df["umur"], bins=[0, 18, 35, 60, 100],
                             labels=["anak", "muda", "dewasa", "lansia"])

# Agregasi (mis. rata-rata transaksi per pelanggan)
agg = df.groupby("pelanggan_id")["transaksi"].agg(["mean", "sum", "count"])
```

> **Mindset:** tanyakan "informasi apa yang akan dipakai manusia ahli untuk memutuskan?" lalu buat fitur yang menangkapnya.

---

## 7. Feature Selection

Fitur terlalu banyak → overfitting, lambat, sulit ditafsir. Kurangi ke yang informatif.

```python
# 1. Buang fitur variansi ~0 (hampir konstan)
from sklearn.feature_selection import VarianceThreshold

# 2. Buang fitur sangat berkorelasi satu sama lain (redundan)
corr = df.corr().abs()
# buang salah satu dari pasangan dengan korelasi > 0.95

# 3. Pilih berdasarkan hubungan dengan target
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# 4. Feature importance dari model tree
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.plot(kind="barh")
```

---

## 8. Pipeline — cara BENAR menyatukan semuanya

`Pipeline` + `ColumnTransformer` mencegah leakage dan membuat kode reproducible. **Ini standar profesional.**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

num_cols = ["umur", "gaji"]
cat_cols = ["kota", "pekerjaan"]

# Pipeline untuk fitur numerik
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Pipeline untuk fitur kategorikal
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Gabungkan per tipe kolom
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# Pipeline lengkap: preprocessing + model
model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)        # semua fit di train saja — anti-leakage otomatis
preds = model.predict(X_test)      # preprocessing diterapkan konsisten
```

**Keuntungan pipeline:**
- Tidak ada leakage (semua `fit` hanya di train).
- Bisa langsung dipakai di `cross_val_score` dan `GridSearchCV`.
- Satu objek untuk disimpan & deploy (Modul 07).

---

## Ringkasan
1. **EDA dulu** — pahami sebelum memodelkan.
2. Tangani **missing & outlier** dengan strategi yang sesuai konteks.
3. **Encode** kategori; **scale** numerik (kecuali tree-based).
4. **Buat fitur** dari pengetahuan domain — sering ini pembeda terbesar.
5. **Selalu pakai Pipeline** untuk mencegah leakage & reproducibility.

## Latihan
1. Ambil dataset Titanic. Lakukan EDA lengkap: missing, distribusi, korelasi survival dengan tiap fitur. Buat minimal 3 visualisasi.
2. Buat 3 fitur baru yang masuk akal (mis. `ukuran_keluarga = sibsp + parch + 1`, `gelar` dari nama, `sendirian`).
3. Bangun `Pipeline` lengkap (imputasi + encoding + scaling + model). Latih & evaluasi.
4. Bandingkan performa model **dengan vs tanpa** feature engineering buatanmu. Berapa peningkatannya?

➡️ Lanjut: [Modul 03 · Classical ML](../03-classical-ml/README.md)
