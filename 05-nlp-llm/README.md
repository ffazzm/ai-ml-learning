# Modul 05 · NLP & LLM / Generative AI

> Dari menghitung kata sampai GPT. Modul ini menelusuri evolusi NLP: text processing klasik → embedding → **Transformer** → **Large Language Models** → cara memakainya dalam aplikasi nyata (**RAG, fine-tuning, prompt engineering**). Inilah area paling panas di AI saat ini.

## Tujuan Belajar
- Memproses & merepresentasikan teks (tokenisasi, TF-IDF, embedding).
- Memahami arsitektur **Transformer** dan mekanisme **attention**.
- Mengerti cara kerja LLM (pretraining, fine-tuning, inference).
- Membangun aplikasi: klasifikasi teks, **semantic search**, **RAG**, chatbot.
- Menguasai **prompt engineering** & memakai API LLM.

## Daftar Isi
1. Text Preprocessing
2. Representasi teks: BoW → TF-IDF → Embedding
3. Word Embeddings (Word2Vec, GloVe)
4. Dari RNN ke Transformer
5. Attention & arsitektur Transformer
6. Large Language Models (LLM)
7. Memakai model pretrained (Hugging Face)
8. Prompt Engineering
9. RAG (Retrieval-Augmented Generation)
10. Fine-tuning

---

## 1. Text Preprocessing

Teks mentah perlu dibersihkan & dipecah jadi unit (token).

```python
text = "Belajar AI/ML itu MENYENANGKAN! Kunjungi https://contoh.com :)"

# Langkah umum (tergantung kebutuhan model):
# - lowercase, hapus URL/punctuation, tokenisasi
import re
text = text.lower()
text = re.sub(r"https?://\S+", "", text)        # hapus URL
text = re.sub(r"[^a-z0-9\s]", "", text)          # hapus simbol
tokens = text.split()                            # tokenisasi sederhana
# -> ['belajar', 'aiml', 'itu', 'menyenangkan', 'kunjungi']
```

Konsep: **tokenisasi** (pecah jadi kata/subword), **stopwords** (buang kata umum "yang/dan/di"), **stemming/lemmatization** (akar kata). Untuk model modern (Transformer), tokenisasi memakai **subword** (BPE/WordPiece) — jangan terlalu agresif membersihkan.

---

## 2. Representasi Teks: dari Hitungan ke Makna

Model butuh angka. Bagaimana mengubah kalimat → vektor?

### Bag of Words (BoW)
Hitung frekuensi tiap kata. Abaikan urutan.

### TF-IDF (Term Frequency–Inverse Document Frequency)
BoW yang dibobot: kata yang sering di satu dokumen tapi jarang di seluruh korpus → lebih informatif. Baseline kuat untuk klasifikasi teks.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

docs = ["saya suka film ini", "film membosankan", "aktingnya luar biasa"]
y = [1, 0, 1]

clf = make_pipeline(TfidfVectorizer(), LogisticRegression())
clf.fit(docs, y)
clf.predict(["film ini bagus"])
```

> 💡 TF-IDF + LogisticRegression masih jadi baseline klasifikasi teks yang sangat kompetitif & murah. Jangan langsung loncat ke LLM untuk masalah sederhana.

**Kelemahan BoW/TF-IDF:** tidak paham makna/sinonim. "mobil" dan "kendaraan" dianggap tak berhubungan. Solusinya: embedding.

---

## 3. Word Embeddings

Petakan kata → vektor padat (mis. 300 dimensi) di mana **kata bermakna mirip berdekatan**. Dipelajari dari konteks (kata yang muncul di lingkungan serupa punya makna serupa).

Sifat ajaibnya — aritmetika makna:
```
vektor("raja") - vektor("pria") + vektor("wanita") ≈ vektor("ratu")
```

```python
# Embedding kalimat modern (siap pakai) dengan sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

emb = model.encode(["kucing duduk di tikar", "anjing tidur di lantai"])
emb.shape          # (2, 384)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([emb[0]], [emb[1]])   # kemiripan semantik
```

Embedding modern berasal dari Transformer dan menangkap makna **kontekstual** (kata "bisa" berbeda di "ular bisa" vs "bisa pergi").

---

## 4. Dari RNN ke Transformer

- **RNN/LSTM** (lama): memproses teks kata-per-kata secara berurutan, menyimpan "memori". Masalah: lambat (tak paralel) & lupa konteks jauh.
- **Transformer** (2017, "Attention is All You Need"): memproses seluruh urutan **sekaligus** lewat **attention**. Paralel, menangkap dependensi jauh. **Fondasi semua LLM modern** (GPT, BERT, Claude, Llama).

---

## 5. Attention & Arsitektur Transformer

### Inti: Self-Attention
Tiap kata "memperhatikan" semua kata lain untuk membangun representasi yang sadar konteks. Untuk tiap token dibuat tiga vektor: **Query (Q)**, **Key (K)**, **Value (V)**.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Intuisi: skor $QK^T$ mengukur seberapa relevan tiap kata terhadap kata lain; softmax mengubahnya jadi bobot; lalu rata-rata tertimbang dari Value. **Multi-head attention** menjalankan ini paralel beberapa kali untuk menangkap beragam jenis hubungan.

### Komponen Transformer
```
Input → Token Embedding + Positional Encoding
      → [ Multi-Head Self-Attention → Add&Norm
          Feed-Forward Network      → Add&Norm ]  × N layer
      → Output
```
- **Positional encoding** — karena attention tak punya urutan bawaan, kita injeksi info posisi.
- **Residual connection + LayerNorm** — bikin jaringan dalam bisa dilatih.

### Tiga varian
| Tipe | Contoh | Untuk |
|---|---|---|
| **Encoder-only** | BERT | pemahaman (klasifikasi, NER, search) |
| **Decoder-only** | GPT, Claude, Llama | generasi teks (chatbot, autocomplete) |
| **Encoder-Decoder** | T5, BART | translasi, ringkasan |

---

## 6. Large Language Models (LLM)

LLM = Transformer (biasanya decoder-only) raksasa yang dilatih memprediksi **token berikutnya** pada teks internet dalam jumlah masif.

**Tahap pembuatan:**
1. **Pretraining** — belajar prediksi token berikutnya dari triliunan token (self-supervised). Menghasilkan "base model" yang berpengetahuan luas. Sangat mahal.
2. **Supervised Fine-Tuning (SFT)** — diajari mengikuti instruksi dari contoh percakapan.
3. **RLHF / alignment** — disetel agar membantu, jujur, aman lewat feedback manusia.

**Konsep kunci memakai LLM:**
- **Token** — unit teks (~¾ kata). Biaya & limit dihitung per token.
- **Context window** — berapa token bisa "dilihat" sekaligus (mis. 200K).
- **Temperature** — kontrol keacakan (0 = deterministik/faktual, tinggi = kreatif).
- **Inference** — menghasilkan token satu per satu (autoregressive).

---

## 7. Memakai Model Pretrained (Hugging Face)

`transformers` memberi akses ribuan model siap pakai.

```python
from transformers import pipeline

# Analisis sentimen (zero setup)
sentiment = pipeline("sentiment-analysis")
sentiment("Materi belajar ini sangat membantu!")
# [{'label': 'POSITIVE', 'score': 0.999}]

# Ringkasan, NER, QA, generasi — semua tersedia
summarizer = pipeline("summarization")
ner = pipeline("ner", aggregation_strategy="simple")
qa = pipeline("question-answering")
```

### Memanggil API LLM (pola umum)
```python
# Contoh pola (sesuaikan dengan provider; pakai prompt caching bila tersedia)
from anthropic import Anthropic
client = Anthropic()
resp = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Jelaskan attention dalam 3 kalimat."}],
)
print(resp.content[0].text)
```

---

## 8. Prompt Engineering

Cara bertanya menentukan kualitas jawaban. Teknik utama:

| Teknik | Inti |
|---|---|
| **Instruksi jelas + peran** | "Kamu ahli X. Lakukan Y dengan format Z." |
| **Few-shot** | beri beberapa contoh input→output |
| **Chain-of-Thought** | "pikirkan langkah demi langkah" → penalaran lebih baik |
| **Structured output** | minta JSON/format spesifik |
| **Context/grounding** | sediakan dokumen relevan dalam prompt (dasar RAG) |

```text
# Buruk
"Tulis tentang anjing."

# Baik
"Kamu penulis sains. Tulis 3 paragraf untuk pembaca awam tentang
bagaimana anjing mengenali emosi manusia. Sertakan 1 studi nyata.
Gunakan bahasa Indonesia yang ringan."
```

---

## 9. RAG (Retrieval-Augmented Generation) ⭐

Masalah: LLM tak tahu data internal/terbaru perusahaanmu dan bisa "berhalusinasi". Solusi paling praktis: **ambil dokumen relevan, lalu suapkan ke LLM sebagai konteks.**

```
Pertanyaan user
      │
      ▼
[Embed pertanyaan] ──► cari di Vector DB ──► top-k dokumen relevan
      │                                              │
      ▼                                              ▼
      └────────►  Prompt = pertanyaan + dokumen  ──► LLM ──► Jawaban
                  (grounded pada data nyata)
```

Pipeline minimal:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Indexing (sekali): pecah dokumen jadi chunk, embed, simpan
docs = ["Kebijakan cuti: 12 hari/tahun.", "Jam kerja: 09.00-17.00.", ...]
doc_emb = embedder.encode(docs)        # di produksi: simpan di FAISS/Chroma/pgvector

# 2. Retrieval: embed pertanyaan, cari chunk paling mirip
q = "berapa jatah cuti saya?"
q_emb = embedder.encode([q])
scores = doc_emb @ q_emb[0]
top = docs[int(np.argmax(scores))]

# 3. Generation: berikan konteks ke LLM
prompt = f"Konteks:\n{top}\n\nPertanyaan: {q}\nJawab hanya berdasarkan konteks."
# -> kirim prompt ke LLM
```

**Komponen produksi:** vector database (FAISS, Chroma, Pinecone, pgvector), strategi **chunking**, **re-ranking**, dan evaluasi (apakah jawaban grounded?). RAG adalah skill paling dicari untuk "AI Engineer" saat ini.

---

## 10. Fine-tuning

Kapan fine-tune vs RAG vs prompting?

| Kebutuhan | Solusi |
|---|---|
| Tambah pengetahuan faktual/terbaru | **RAG** |
| Ubah gaya/format/perilaku konsisten | **Fine-tuning** |
| Tugas umum, butuh cepat | **Prompt engineering** |

**Fine-tuning efisien — LoRA/PEFT:** alih-alih melatih ulang miliaran parameter, latih sedikit "adapter" parameter → murah & cepat.

```python
# Konsep (pakai peft + transformers):
# 1. Load base model
# 2. Bungkus dengan LoRA config (rank kecil, target attention layers)
# 3. Train pada dataset instruksi domain-mu
# 4. Simpan adapter (beberapa MB, bukan GB)
```

---

## Ringkasan
1. Teks → angka: TF-IDF (klasik) → embedding (semantik).
2. **Transformer + attention** adalah fondasi semua LLM.
3. LLM dilatih prediksi token berikutnya, lalu di-align.
4. Untuk aplikasi: **prompting** → **RAG** → **fine-tuning** sesuai kebutuhan.
5. Mulai dari yang sederhana & murah; naik kompleksitas hanya bila perlu.

## Latihan
1. Bangun classifier sentimen dengan TF-IDF + LogisticRegression. Bandingkan dengan `pipeline("sentiment-analysis")` Hugging Face.
2. Buat **semantic search**: embed 50 kalimat, cari yang paling mirip dengan query. Bandingkan hasilnya dengan pencarian kata kunci biasa.
3. Bangun **mini-RAG** atas 5–10 dokumen teks (mis. FAQ). Uji apakah jawaban benar-benar grounded.
4. Eksperimen prompt engineering: bandingkan jawaban zero-shot vs few-shot vs chain-of-thought untuk soal penalaran.
5. (Lanjut) Fine-tune model kecil dengan LoRA pada dataset domain pilihanmu.

➡️ Lanjut: [Modul 06 · Computer Vision](../06-computer-vision/README.md)
