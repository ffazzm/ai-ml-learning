# Modul 09 · Evaluasi & Interpretasi Model (Post-Training)

> Model sudah dilatih, akurasinya 87%. **Lalu apa?** "87%" tidak memberitahu apa pun yang berguna sampai kamu bisa menjawab: *kenapa segitu? salahnya di mana? bisakah dipercaya? bagaimana cara memperbaikinya?* Modul ini soal **mendiagnosis & menjelaskan** model setelah training — skill yang sering diabaikan tapi paling dicari di kerja nyata.

## Tujuan Belajar
- Mendiagnosis **kenapa** performa model seperti itu (bukan cuma mengukurnya).
- Melakukan **error analysis** — melihat kesalahan model secara sistematis.
- Membaca **learning curve** untuk memutuskan: butuh data lebih banyak? model lebih kompleks?
- Memeriksa apakah probabilitas model bisa dipercaya (**calibration**).
- Menjelaskan keputusan model secara **global** (feature importance, PDP) & **lokal** (SHAP, LIME).
- Menjelaskan model **deep learning** (Grad-CAM, attention).
- Mengevaluasi keadilan/robustness antar **subkelompok** (slicing).

## Daftar Isi
1. Kerangka berpikir: dari "berapa" ke "kenapa"
2. Membedah confusion matrix & metrik per-kelas
3. Error analysis — lihat kesalahannya
4. Learning curve & validation curve (diagnosis bias/variance)
5. Calibration — apakah confidence-nya jujur?
6. Interpretabilitas GLOBAL (perilaku model keseluruhan)
7. Interpretabilitas LOKAL (kenapa prediksi INI begini) → SHAP & LIME
8. Menjelaskan model deep learning
9. Slicing & analisis subkelompok
10. Alur kerja debugging: "kenapa akurasi saya begini?"

---

## 1. Kerangka Berpikir: dari "Berapa" ke "Kenapa"

Satu angka akurasi menyembunyikan semua cerita penting. Selalu pecah pertanyaan jadi:

| Pertanyaan | Alat |
|---|---|
| Salahnya **di mana** (kelas/grup mana)? | confusion matrix, metrik per-kelas, slicing |
| Salahnya **seperti apa** (pola kesalahan)? | error analysis manual |
| Apakah masalahnya **bias atau variance**? | learning curve, gap train-val |
| Apakah **confidence**-nya bisa dipercaya? | calibration curve |
| **Fitur apa** yang menggerakkan model? | feature/permutation importance, PDP/SHAP |
| **Kenapa prediksi spesifik ini** begitu? | SHAP/LIME lokal |
| Apakah model **adil & robust** di semua grup? | analisis subkelompok |

> Prinsip: **akurasi adalah gejala, bukan diagnosis.** Tugasmu mencari akar penyebabnya.

---

## 2. Membedah Confusion Matrix & Metrik Per-Kelas

Akurasi global menipu, terutama saat kelas tidak seimbang (lihat Modul 01). Mulai selalu dari sini.

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.show()

print(classification_report(y_test, y_pred, target_names=class_names))
```

**Cara membacanya — pertanyaan yang harus kamu ajukan:**
- Kelas mana yang **recall**-nya rendah? → model sering *melewatkan* kelas itu.
- Kelas mana yang **precision**-nya rendah? → model sering *salah alarm* untuk kelas itu.
- Sel off-diagonal mana yang **paling besar**? → dua kelas yang sering **tertukar** (mungkin memang mirip, atau labelnya ambigu).

> Contoh diagnosis: "Akurasi 87% tapi recall kelas 'fraud' cuma 30%." → Modelmu sebenarnya buruk untuk tujuan utamanya; angka 87% didongkrak kelas mayoritas. Inilah kenapa kita tidak berhenti di akurasi.

---

## 3. Error Analysis — Lihat Kesalahannya

Teknik paling underrated & paling berdampak: **baca contoh-contoh yang salah diprediksi satu per satu.** Manusia hebat menemukan pola.

```python
import numpy as np
import pandas as pd

# Kumpulkan kasus salah, urutkan dari yang model paling "pede" tapi salah
proba = model.predict_proba(X_test)
confidence = proba.max(axis=1)
salah = y_pred != y_test

errors = pd.DataFrame({
    "index": np.arange(len(y_test))[salah],
    "aktual": y_test[salah],
    "prediksi": y_pred[salah],
    "confidence": confidence[salah],
}).sort_values("confidence", ascending=False)   # salah TAPI sangat yakin = paling informatif

print(errors.head(20))
# Lalu BUKA data mentah untuk index ini dan tanya: kenapa model salah?
```

**Pola yang biasa ditemukan & artinya:**
| Temuan dari membaca error | Akar masalah & tindakan |
|---|---|
| Banyak label aktualnya sebenarnya salah/ambigu | **noise di label** → bersihkan data, bukan utak-atik model |
| Salah selalu di subkelompok tertentu | **underrepresentation** → tambah data grup itu |
| Salah pada input yang "aneh"/di luar distribusi | training data tak mewakili produksi |
| Salah karena fitur kunci hilang/keliru | **feature engineering** kurang (Modul 02) |
| Confidence tinggi tapi salah | model **overconfident** → cek calibration (bagian 5) |

> Aturan praktis industri (Andrew Ng): luangkan waktu memeriksa ~100 contoh salah secara manual sebelum memutuskan langkah berikutnya. Hampir selalu menghemat berminggu-minggu eksperimen buta.

---

## 4. Learning Curve & Validation Curve (Diagnosis Bias/Variance)

Ini yang menjawab langsung: **"kenapa akurasinya cuma segini, dan apa solusinya?"**

### Learning curve — plot performa vs jumlah data training

```python
from sklearn.model_selection import learning_curve
import numpy as np, matplotlib.pyplot as plt

sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

plt.plot(sizes, train_scores.mean(1), "o-", label="train")
plt.plot(sizes, val_scores.mean(1), "o-", label="validation")
plt.xlabel("jumlah sampel training"); plt.ylabel("skor"); plt.legend(); plt.show()
```

**Cara mendiagnosis dari bentuk kurva:**

```
HIGH BIAS (underfit)          HIGH VARIANCE (overfit)        PAS
skor│ train ──────            skor│ train ───────            skor│ train ─────
    │ val ─────── (rapat,         │           ╲ (gap besar)      │ val ──────  (rapat,
    │            keduanya         │ val ──╱                      │            keduanya
    │            RENDAH)          │                              │            TINGGI)
    └──────── data →             └──────── data →               └──────── data →

→ Model terlalu sederhana.     → Model menghafal.             → Sudah baik.
  Tambah fitur/kompleksitas,     Tambah DATA, regularisasi,     Tambah data hanya
  kurangi regularisasi.          kurangi kompleksitas.          beri sedikit gain.
  (Tambah data TIDAK menolong.)  (Gap menutup dgn data.)
```

> Ini menjawab pertanyaan mahal: **"haruskah saya kumpulkan lebih banyak data?"** Kalau high-bias → percuma, perbaiki modelnya dulu. Kalau high-variance → ya, data membantu.

### Validation curve — performa vs satu hyperparameter

```python
from sklearn.model_selection import validation_curve
# mis. lihat efek max_depth pada train vs val -> temukan titik mulai overfit
param_range = [1, 2, 4, 8, 16, 32]
train_s, val_s = validation_curve(model, X, y, param_name="max_depth",
                                  param_range=param_range, cv=5)
```
Titik di mana skor train terus naik tapi skor val mulai turun = ambang overfitting → pilih hyperparameter di situ.

---

## 5. Calibration — Apakah Confidence-nya Jujur?

Saat model bilang "yakin 90%", apakah benar-benar benar 90% dari waktu? Model (terutama boosting & neural net) sering **overconfident**. Penting jika kamu memakai probabilitasnya untuk keputusan (threshold, ranking, risiko).

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, proba[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true, "o-", label="model")
plt.plot([0, 1], [0, 1], "--", label="kalibrasi sempurna")
plt.xlabel("confidence yang diprediksi"); plt.ylabel("frekuensi benar sesungguhnya")
plt.legend(); plt.show()

# Perbaiki kalibrasi (Platt scaling / isotonic) jika melenceng dari garis diagonal:
calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5).fit(X_train, y_train)
```
Kurva di **bawah** diagonal = overconfident. Di **atas** = underconfident.

---

## 6. Interpretabilitas GLOBAL — Perilaku Model Keseluruhan

"Fitur apa yang umumnya digerakkan model, dan ke arah mana?"

### a) Feature importance (model tree)
Cepat, tapi hati-hati: bias ke fitur berkardinalitas tinggi.
```python
import pandas as pd
pd.Series(model.feature_importances_, index=feature_names).sort_values().plot.barh()
```

### b) Permutation importance (lebih andal, model apa pun)
Acak satu kolom, ukur seberapa turun performanya. Turun banyak = fitur itu penting.
```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
pd.Series(r.importances_mean, index=feature_names).sort_values().plot.barh()
```

### c) Partial Dependence Plot (PDP) — *arah* pengaruh fitur
Menunjukkan bagaimana prediksi berubah saat satu fitur naik (rata-rata fitur lain). Menjawab: "apakah naiknya umur menaikkan atau menurunkan prediksi, dan polanya linear/tidak?"
```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X_test, features=["umur", "gaji"])
```

---

## 7. Interpretabilitas LOKAL — Kenapa Prediksi *INI* Begini? (SHAP & LIME)

Untuk satu prediksi spesifik ("kenapa pinjaman pelanggan X ditolak?"), kamu butuh penjelasan **per-instance**.

### SHAP (paling direkomendasikan) ⭐
Berbasis teori game (Shapley values): membagi kontribusi tiap fitur secara adil terhadap prediksi. Konsisten secara global maupun lokal.

```python
import shap   # pip install shap

explainer = shap.TreeExplainer(model)        # ada juga untuk model lain (KernelExplainer, deep)
shap_values = explainer.shap_values(X_test)

# Penjelasan SATU prediksi: kontribusi tiap fitur mendorong naik/turun
shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                     base_values=explainer.expected_value, data=X_test.iloc[0]))

# Ringkasan GLOBAL dari nilai lokal: fitur terpenting + arah pengaruh
shap.summary_plot(shap_values, X_test)
```
SHAP menjawab persis: *"prediksi ini 0.82 karena `gaji` tinggi (+0.3) dan `umur` muda (−0.1)..."* — inilah bentuk "explanation" yang ditanyakan stakeholder/regulator.

### LIME
Membuat model linear sederhana di sekitar satu prediksi untuk menjelaskannya. Lebih cepat dari SHAP tapi kurang stabil. Berguna untuk teks & gambar juga.
```python
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names,
                                 class_names=class_names, mode="classification")
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp.show_in_notebook()
```

| | SHAP | LIME |
|---|---|---|
| Dasar | Shapley values (teori solid) | aproksimasi lokal linear |
| Konsistensi | tinggi | bisa goyah antar-run |
| Kecepatan | lebih lambat | cepat |
| Saran | **default** untuk tabular | cek cepat / teks & gambar |

---

## 8. Menjelaskan Model Deep Learning

Neural network = black box yang lebih pekat. Alatnya:

- **Grad-CAM** (gambar/CNN) — peta panas menunjukkan **piksel mana** yang membuat CNN memutuskan "kucing". Cara cepat mengecek apakah model "melihat" objek yang benar atau malah berpegang pada latar/artefak.
- **Attention visualization** (Transformer/NLP) — token mana yang "diperhatikan" model saat memprediksi (lihat attention, Modul 05).
- **Saliency / Integrated Gradients** — gradien output terhadap input → fitur mana paling berpengaruh.
- **SHAP DeepExplainer** — versi SHAP untuk jaringan dalam.

```python
# Pola Grad-CAM (mis. pytorch-grad-cam): hasilkan heatmap, tumpuk di atas gambar
# -> verifikasi visual: "apakah model fokus ke objek atau ke noise?"
```
> Kasus klasik: model "deteksi serigala vs husky" ternyata cuma mendeteksi **salju di latar**. Grad-CAM membongkar ini — tanpa interpretasi, kamu deploy model yang salah belajar.

---

## 9. Slicing & Analisis Subkelompok

Performa rata-rata menyembunyikan kegagalan pada grup spesifik. Evaluasi per-irisan data.

```python
# Hitung metrik per subkelompok (mis. per wilayah/gender/rentang umur)
for grup, idx in X_test.groupby("wilayah").groups.items():
    acc = (y_pred[idx] == y_test.loc[idx]).mean()
    print(f"{grup}: akurasi {acc:.3f}")
```
Tujuan: menemukan grup di mana model jauh lebih buruk → masalah **keadilan (fairness)**, **bias data**, atau **robustness**. Sering jadi temuan terpenting sebelum produksi (Modul 07).

---

## 10. Alur Kerja Debugging: "Kenapa Akurasi Saya Begini?"

Checklist berurutan saat performa mengecewakan:

```
1. Apakah evaluasinya benar?      → cek leakage (Modul 01), metrik tepat, split benar.
2. Bandingkan dengan baseline.    → vs tebakan kelas mayoritas / model sederhana. Lebih baik?
3. Bias atau variance?            → learning curve (bagian 4).
       ├─ High bias  → fitur/model lebih kuat, kurangi regularisasi.
       └─ High variance → lebih banyak data, regularisasi, model lebih sederhana.
4. Di mana salahnya?              → confusion matrix + metrik per-kelas (bagian 2).
5. Seperti apa salahnya?          → error analysis manual ~100 kasus (bagian 3).
       ├─ Label noise → bersihkan data.
       ├─ Fitur kurang → feature engineering (Modul 02).
       └─ Grup tertentu → tambah data grup itu (bagian 9).
6. Apa yang digerakkan model?     → SHAP/permutation importance (bagian 6-7).
       └─ Kalau fitur "aneh" dominan → kemungkinan leakage/spurious correlation.
7. Apakah confidence dipercaya?   → calibration (bagian 5).
8. Iterasi — ubah SATU hal, ukur lagi.
```

> **Mindset inti:** model yang "jelek" jarang butuh algoritma yang lebih keren. Ia butuh kamu **memahami kesalahannya**. Diagnosis dulu, baru obati.

---

## Ringkasan
1. Akurasi = gejala. Selalu cari **kenapa** lewat confusion matrix, error analysis, learning curve.
2. **Learning curve** memberitahu apakah masalahnya bias (perbaiki model) atau variance (tambah data).
3. **Calibration** memastikan probabilitas bisa dipercaya.
4. **Global** (importance, PDP) vs **lokal** (SHAP, LIME) menjelaskan perilaku model.
5. Untuk deep learning pakai **Grad-CAM / attention**.
6. **Slicing** membongkar kegagalan tersembunyi per subkelompok.

## Latihan
1. Latih classifier pada dataset tak seimbang. Tunjukkan bahwa akurasi tinggi tapi recall kelas minoritas rendah lewat confusion matrix.
2. Buat learning curve untuk model sederhana vs kompleks pada data sama. Diagnosis mana yang high-bias, mana yang high-variance.
3. Lakukan error analysis: ambil 30 prediksi salah dengan confidence tertinggi, baca datanya, dan tulis 3 hipotesis penyebab.
4. Pasang SHAP pada model tabular. Jelaskan satu prediksi spesifik dengan waterfall plot dalam kalimat biasa.
5. Plot calibration curve. Apakah modelmu overconfident? Perbaiki dengan `CalibratedClassifierCV` dan bandingkan.
6. (CV) Pasang Grad-CAM pada classifier gambar dan verifikasi model fokus ke objek yang benar.

⬅️ Kembali ke [Daftar Modul](../README.md) · Terkait: [Modul 01 · Fondasi ML](../01-fondasi-ml/README.md) (metrik) & [Modul 07 · MLOps](../07-mlops-deployment/README.md) (monitoring produksi)
