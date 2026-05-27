# Modul 03 · Classical Machine Learning

> Sebelum deep learning, kuasai algoritma "klasik". Untuk **data tabular** (kebanyakan masalah bisnis nyata), model klasik seperti **XGBoost** sering **mengalahkan** neural network — lebih cepat, lebih mudah ditafsir, butuh data lebih sedikit. Ini roti & mentega seorang ML Engineer.

## Tujuan Belajar
- Memahami & mengimplementasikan algoritma supervised utama.
- Tahu kapan memakai algoritma mana.
- Menguasai **ensemble** (Random Forest, Gradient Boosting) yang mendominasi kompetisi data tabular.
- Memahami unsupervised: **clustering** & **PCA**.
- Melakukan **hyperparameter tuning**.

## Daftar Isi
1. Regresi Linear (dengan implementasi dari nol → [linear_regression_scratch.py](./linear_regression_scratch.py))
2. Regresi Logistik
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)
6. Decision Tree
7. Ensemble: Random Forest & Gradient Boosting
8. Unsupervised: K-Means & PCA
9. Hyperparameter Tuning

---

## 1. Regresi Linear

Model paling fundamental. Memprediksi nilai kontinu sebagai kombinasi linear fitur:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

**Loss (MSE):** $L = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$. Kita cari $\mathbf{w}, b$ yang meminimalkannya — lewat gradient descent (Modul 00.3) atau closed-form.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression().fit(X_train, y_train)
print("R²:", r2_score(y_test, model.predict(X_test)))
print("Koefisien:", model.coef_, "Intercept:", model.intercept_)
```

> 📂 **Implementasi dari nol** ada di [linear_regression_scratch.py](./linear_regression_scratch.py) — wajib baca & jalankan untuk benar-benar paham gradient descent bekerja.

### Regularisasi — mencegah overfitting
Tambahkan penalti untuk bobot besar:
- **Ridge (L2):** $L + \alpha \sum w_i^2$ — mengecilkan bobot, tidak ke nol.
- **Lasso (L1):** $L + \alpha \sum |w_i|$ — bisa membuat bobot **tepat nol** (feature selection otomatis).
- **ElasticNet:** kombinasi keduanya.

```python
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)   # lihat berapa koefisien jadi 0
```

---

## 2. Regresi Logistik (klasifikasi, bukan regresi!)

Untuk **klasifikasi**. Melewatkan output linear ke fungsi **sigmoid** agar jadi probabilitas [0,1]:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \hat{p} = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

Loss-nya **cross-entropy** (Modul 00.4). Prediksi kelas 1 jika $\hat{p} > 0.5$.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
clf.predict(X_test)              # label 0/1
clf.predict_proba(X_test)        # probabilitas tiap kelas
```

> Meski sederhana, ini **baseline wajib** untuk klasifikasi. Cepat, interpretable, dan sering cukup baik. Selalu mulai dari sini sebelum model rumit.

---

## 3. K-Nearest Neighbors (KNN)

"Kamu adalah rata-rata 5 tetanggamu." Untuk prediksi, lihat **k titik terdekat** dan ambil mayoritas (klasifikasi) atau rata-rata (regresi).

- Tidak ada "training" — semua kerja saat prediksi (lazy learning).
- **Wajib scaling** (berbasis jarak).
- Lambat untuk data besar; sensitif terhadap dimensi tinggi (curse of dimensionality).

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
```

---

## 4. Naive Bayes

Berbasis Teorema Bayes (Modul 00.4) dengan asumsi "naif" bahwa fitur **independen**. Meski asumsinya jarang benar, ia bekerja sangat baik untuk **klasifikasi teks** (spam, sentimen) dan sangat cepat.

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB
nb = MultinomialNB().fit(X_train, y_train)   # untuk fitur hitungan (teks)
```

---

## 5. Support Vector Machine (SVM)

Mencari **hyperplane** yang memisahkan kelas dengan **margin** terbesar. Dengan **kernel trick** (mis. RBF), bisa menangani batas non-linear.

- Kuat untuk data berdimensi tinggi, dataset kecil-menengah.
- **Wajib scaling**. Lambat untuk data sangat besar.

```python
from sklearn.svm import SVC
svm = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_train_scaled, y_train)
```

---

## 6. Decision Tree

Serangkaian pertanyaan "ya/tidak" yang membagi data. Memilih split yang paling mengurangi **impurity** (Gini/entropi).

- ✅ Sangat interpretable, tak butuh scaling, tangani non-linear.
- ❌ Mudah overfit (pohon dalam menghafal data).

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)

import matplotlib.pyplot as plt
plot_tree(tree, feature_names=X.columns, filled=True); plt.show()
```

Parameter pengontrol overfitting: `max_depth`, `min_samples_leaf`, `min_samples_split`.

---

## 7. Ensemble Methods ⭐ (yang dipakai di dunia nyata)

Gabungkan banyak model lemah → satu model kuat. **Ini juara untuk data tabular.**

### Bagging — Random Forest
Latih banyak tree pada **subset data + subset fitur acak**, lalu voting. Mengurangi variance/overfitting dari tree tunggal.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                            n_jobs=-1, random_state=42).fit(X_train, y_train)
# Bonus: feature importance gratis
import pandas as pd
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot.barh()
```

### Boosting — Gradient Boosting / XGBoost / LightGBM
Latih tree **berurutan**, tiap tree memperbaiki kesalahan sebelumnya. Biasanya **akurasi tertinggi** untuk data tabular.

```python
# XGBoost — pemenang banyak kompetisi Kaggle
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
).fit(X_train, y_train)

# LightGBM — lebih cepat untuk dataset besar
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05).fit(X_train, y_train)
```

| | Random Forest (Bagging) | Gradient Boosting |
|---|---|---|
| Cara | tree paralel & independen | tree berurutan, saling koreksi |
| Kekuatan | robust, sedikit tuning, anti-overfit | akurasi puncak |
| Risiko | sedikit di bawah boosting | overfit jika tuning buruk |
| Saran | baseline kuat | model utama untuk performa |

> 💡 **Resep praktis untuk data tabular:** mulai LogisticRegression (baseline) → RandomForest (cek cepat) → XGBoost/LightGBM (tuning untuk performa final).

---

## 8. Unsupervised Learning

### K-Means Clustering
Kelompokkan data ke **k cluster**: ulangi (1) assign tiap titik ke centroid terdekat, (2) pindahkan centroid ke rata-rata anggotanya.

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
labels = km.labels_

# Pilih k dengan "elbow method" (plot inertia vs k) atau silhouette score
from sklearn.metrics import silhouette_score
silhouette_score(X_scaled, labels)
```

### PCA (Principal Component Analysis)
Reduksi dimensi: proyeksikan data ke arah variansi terbesar (pakai SVD/eigen, Modul 00.2). Untuk visualisasi, kompresi, dan mengatasi multikolinearitas.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)            # turunkan ke 2D untuk plot
X_2d = pca.fit_transform(X_scaled)
print("Variansi dijelaskan:", pca.explained_variance_ratio_)
```

---

## 9. Hyperparameter Tuning

Hyperparameter = setelan yang **tidak** dipelajari model (mis. `max_depth`, `learning_rate`). Cari kombinasi terbaik secara sistematis dengan cross-validation.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
}
search = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)

print("Param terbaik:", search.best_params_)
print("Skor CV terbaik:", search.best_score_)
best_model = search.best_estimator_
```

- **GridSearchCV:** coba semua kombinasi (lengkap tapi mahal).
- **RandomizedSearchCV:** sampel acak (lebih efisien untuk ruang besar).
- Lanjutan: **Optuna** (Bayesian optimization) — lebih pintar & cepat.

---

## Cheat Sheet: Memilih Algoritma

| Masalah | Coba dulu |
|---|---|
| Regresi tabular | LinearRegression → RandomForest → XGBoost |
| Klasifikasi tabular | LogisticRegression → RandomForest → XGBoost/LightGBM |
| Teks (spam/sentimen) | Naive Bayes / LogisticRegression + TF-IDF |
| Dataset kecil, dimensi tinggi | SVM |
| Butuh interpretasi tinggi | DecisionTree / LogisticRegression |
| Tanpa label, segmentasi | K-Means |
| Terlalu banyak fitur | PCA + model |

## Latihan
1. Jalankan [linear_regression_scratch.py](./linear_regression_scratch.py), pahami tiap baris. Plot turunnya loss.
2. Pada dataset klasifikasi (mis. Breast Cancer dari `sklearn.datasets`), bandingkan 5 algoritma di atas. Buat tabel akurasi/F1.
3. Tuning XGBoost dengan GridSearchCV. Berapa peningkatan vs default?
4. Pada dataset tanpa label, jalankan K-Means + PCA. Visualisasikan cluster dalam 2D.
5. Jelaskan: kenapa Random Forest jarang overfit padahal tiap tree-nya bisa overfit?

➡️ Lanjut: [Modul 04 · Deep Learning](../04-deep-learning/README.md)
