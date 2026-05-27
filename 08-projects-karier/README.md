# Modul 08 · Projects & Roadmap Karier

> Teori tanpa praktik cepat menguap. Portfolio adalah bukti kemampuanmu — lebih meyakinkan dari sertifikat mana pun. Modul ini berisi ide project bertingkat, persiapan interview, dan peta jalan menjadi AI/ML Engineer.

## Daftar Isi
1. Mengapa project end-to-end penting
2. Ide project bertingkat (pemula → mahir)
3. Anatomi project portfolio yang bagus
4. Roadmap karier
5. Persiapan interview
6. Kebiasaan belajar berkelanjutan

---

## 1. Mengapa Project End-to-End?

Tutorial mengajarkan langkah; project mengajarkan **menyatukan langkah & menghadapi kekacauan nyata** — data kotor, model tak konvergen, deployment gagal. Recruiter & engineer menilai dari ini.

> **Aturan:** untuk tiap modul yang selesai, kerjakan minimal satu project yang memakai materinya. Bangun **portfolio di GitHub** dengan README yang menjelaskan masalah, pendekatan, hasil, dan cara menjalankan.

---

## 2. Ide Project Bertingkat

### 🟢 Pemula (setelah Modul 0–3)
1. **Prediksi harga rumah** — regresi, EDA, feature engineering, evaluasi (dataset: Ames/California Housing).
2. **Prediksi survival Titanic** — klasifikasi end-to-end + pipeline.
3. **Segmentasi pelanggan** — K-Means + PCA pada data ritel.
4. **Deteksi spam SMS** — TF-IDF + Naive Bayes/LogReg.

### 🟡 Menengah (setelah Modul 4–6)
5. **Klasifikasi gambar** dengan transfer learning (mis. jenis bunga/makanan) + deploy sebagai API.
6. **Analisis sentimen** ulasan produk dengan model Transformer (fine-tune DistilBERT).
7. **Sistem rekomendasi** film/produk (collaborative filtering).
8. **Forecasting time series** (penjualan/harga) — fitur temporal + model.

### 🔴 Mahir (setelah Modul 7 + LLM)
9. **Chatbot RAG** atas dokumen sendiri (FAQ perusahaan, catatan kuliah) — embedding + vector DB + LLM, deploy dengan UI.
10. **End-to-end MLOps**: training → MLflow → API FastAPI → Docker → cloud → monitoring drift.
11. **Fine-tune LLM dengan LoRA** untuk tugas domain spesifik.
12. **Agen multimodal** — gabungkan vision + bahasa (mis. jawab pertanyaan tentang gambar).

> 💡 **Project terbaik = menyelesaikan masalah yang kamu pedulikan** dengan data nyata. Lebih berkesan dari mengulang dataset standar.

---

## 3. Anatomi Project Portfolio yang Bagus

```
nama-project/
├── README.md          ← masalah, pendekatan, hasil (dengan grafik!), cara jalankan
├── data/              ← atau skrip unduh (jangan commit data besar)
├── notebooks/         ← EDA & eksperimen
├── src/               ← kode bersih & modular (bukan cuma notebook)
├── models/            ← model tersimpan / link
├── requirements.txt
├── Dockerfile         ← bonus besar: tunjukkan kamu bisa deploy
└── tests/             ← bonus: tunjukkan rigor engineering
```

README yang baik menjawab: **Masalah apa? Data dari mana? Pendekatan & kenapa? Hasil (metrik + visual)? Apa yang dipelajari & batasannya?** Tampilkan hasil di awal.

---

## 4. Roadmap Karier

```
Fase 1: FONDASI (2-4 bln)     → Modul 0-3. Python, math, ML klasik. 2-3 project pemula.
Fase 2: DEEP LEARNING (2-3 bln) → Modul 4 + (5 atau 6). PyTorch, satu spesialisasi.
Fase 3: PRODUKSI (1-2 bln)    → Modul 7. Deploy ≥1 model. Pelajari Git, SQL, cloud dasar.
Fase 4: SPESIALISASI & KERJA  → Perdalam NLP/LLM atau CV. Project mahir. Lamar & kontribusi open-source.
```

### Peran di industri (kenali perbedaannya)
| Peran | Fokus |
|---|---|
| **ML Engineer** | bangun & deploy model ke produksi (kode + ML + sistem) |
| **Data Scientist** | analisis, eksperimen, insight bisnis |
| **AI Engineer / LLM Engineer** | aplikasi berbasis LLM (RAG, agent, prompt, integrasi) — peran paling tumbuh |
| **MLOps Engineer** | infrastruktur, pipeline, monitoring |
| **Research Engineer** | implementasi & eksperimen riset mutakhir |

### Skill pendamping yang sering diabaikan (tapi krusial)
- **SQL** — mengambil data hampir selalu lewat SQL.
- **Git/GitHub** — kolaborasi & versioning.
- **Linux & command line.**
- **Cloud dasar** (AWS/GCP/Azure) — minimal satu.
- **Komunikasi** — menjelaskan hasil ke non-teknis. Pembeda besar di level senior.

---

## 5. Persiapan Interview

Interview AI/ML biasanya mencakup:

1. **Coding** — algoritma & struktur data (LeetCode easy-medium), plus manipulasi data Python/Pandas.
2. **ML fundamentals** — bias-variance, overfitting, metrik, regularisasi, cara kerja algoritma (Modul 01 & 03). *Bisa jelaskan tanpa rumus?*
3. **ML system design** — "rancang sistem rekomendasi/deteksi penipuan." Bahas: data, fitur, model, evaluasi, serving, monitoring (seluruh kurikulum ini!).
4. **Project sendiri** — siap diinterogasi dalam tentang keputusanmu: *kenapa model itu? kenapa metrik itu? apa yang akan kamu perbaiki?*
5. **Matematika/teori** (peran riset) — kalkulus, aljabar linear, probabilitas.

> **Tips:** untuk tiap konsep, latih menjelaskannya dalam 2 level — intuisi (untuk PM) dan teknis (untuk engineer). Jika bisa keduanya, kamu benar-benar paham.

---

## 6. Kebiasaan Belajar Berkelanjutan

AI bergerak cepat — yang penting adalah **fondasi kuat + kemampuan belajar terus**, bukan menghafal tren.

- **Implementasi ulang paper/konsep** dari nol sesekali — pemahaman terdalam.
- **Ikuti sumber berkualitas:** blog engineering perusahaan AI, Papers with Code, beberapa peneliti tepercaya. Hindari kebisingan hype.
- **Kaggle** — latihan praktik & belajar dari notebook publik.
- **Ajarkan/tulis** — menulis blog atau menjelaskan ke orang lain memaksa kejelasan.
- **Bangun terus** — satu project kecil > sepuluh tutorial ditonton.

---

## Kata Penutup

Menjadi AI/ML Engineer yang kuat bukan soal tahu setiap algoritma terbaru, tapi:
1. **Fondasi solid** — math, statistik, cara kerja model (Modul 0–4).
2. **Insting rekayasa** — data bersih, evaluasi jujur, kode reproducible, deploy & monitor (Modul 2, 7).
3. **Penilaian** — memilih alat paling sederhana yang menyelesaikan masalah, bukan yang paling keren.
4. **Belajar tanpa henti.**

Kerjakan modul-modulnya, ketik setiap kodenya, bangun project, dan ulangi. **Selamat membangun! 🚀**

⬅️ Kembali ke [Daftar Modul](../README.md)
