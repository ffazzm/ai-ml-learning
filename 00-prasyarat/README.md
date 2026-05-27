# Modul 00 · Prasyarat

> Fondasi yang **tidak boleh dilewati**. AI/ML adalah matematika yang dieksekusi oleh kode. Modul ini membangun dua kaki itu: **Python untuk data** dan **matematika untuk ML**.

## Tujuan Belajar

Setelah modul ini kamu bisa:
- Menulis Python yang idiomatik untuk manipulasi data (NumPy & Pandas).
- Memahami **vektor, matriks, dan operasinya** — bahasa asli ML.
- Mengerti **turunan & gradien** — mesin di balik semua "learning".
- Membaca notasi **probabilitas & statistik** yang muncul di paper dan dokumentasi.

## Daftar Isi

0. [Dasar-Dasar Python (dari Nol)](./00-python-dasar.md) ← mulai di sini kalau baru belajar Python
1. [Python untuk Data Science (NumPy & Pandas)](./01-python-data-science.md)
2. [Aljabar Linear](./02-aljabar-linear.md)
3. [Kalkulus & Gradien](./03-kalkulus-gradien.md)
4. [Probabilitas & Statistik](./04-probabilitas-statistik.md)
5. [Penerapan Matematika & Statistika dalam Proses AI/ML](./05-aplikasi-matematika-di-ml.md) ← peta "konsep mana dipakai di tahap mana"

---

## Mengapa matematika penting (dan seberapa dalam)?

Kamu **tidak perlu** jadi matematikawan. Tapi kamu perlu **intuisi** dan kemampuan membaca notasi. Berikut peta "untuk apa":

| Bidang Matematika | Dipakai untuk |
|---|---|
| **Aljabar Linear** | Representasi data (matriks), operasi neural network, PCA, embedding |
| **Kalkulus** | Gradient descent, backpropagation — inti dari training |
| **Probabilitas** | Naive Bayes, model generatif, loss (cross-entropy), ketidakpastian |
| **Statistik** | Evaluasi model, hipotesis, distribusi data, A/B testing |

### Aturan praktis

> Pelajari konsep sampai kamu bisa **menjelaskannya ke orang lain dengan kata-katamu sendiri** dan **mengimplementasikannya dengan NumPy**. Itu cukup untuk jadi AI/ML Engineer yang kuat.

---

## Checklist Sebelum Lanjut ke Modul 01

- [ ] Bisa membuat & memanipulasi `np.array` (indexing, slicing, broadcasting).
- [ ] Bisa load CSV, filter, group, dan agregasi dengan Pandas.
- [ ] Paham apa itu **dot product** dan **perkalian matriks**, dan kapan dipakai.
- [ ] Paham **gradien** sebagai "arah kenaikan tercuram" dan kaitannya dengan minimisasi loss.
- [ ] Paham **mean, variansi, distribusi normal**, dan **conditional probability** `P(A|B)`.

Kalau semua tercentang, lanjut ke [Modul 01 · Fondasi ML](../01-fondasi-ml/README.md).
