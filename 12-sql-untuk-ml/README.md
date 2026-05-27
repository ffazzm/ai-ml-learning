# Modul 12 · SQL untuk AI/ML Engineer

> Sebelum data sampai ke Pandas atau model, ia hampir selalu hidup di **database**. Mengambil, menggabungkan, dan menyiapkan data lewat SQL adalah skill **harian** ML Engineer — tapi sering diabaikan di kurikulum ML. Modul ini mengajarkan SQL **dari sudut pandang ML**: bukan untuk membangun aplikasi, tapi untuk *menarik data & membuat fitur*.

## Tujuan Belajar
- Menulis query dari dasar sampai menengah dengan percaya diri.
- Menguasai **JOIN** untuk menggabungkan fitur dari banyak tabel.
- Menguasai **window functions** ⭐ — senjata utama untuk fitur agregat & time series tanpa leakage.
- Melakukan **feature engineering langsung di SQL** (lebih efisien dari menarik semua ke Python).
- Menghubungkan SQL → Pandas dengan benar, dan tahu kapan proses di DB vs Python.

## Daftar Isi
1. Kenapa ML Engineer butuh SQL
2. Dasar query (SELECT/WHERE/ORDER)
3. Agregasi (GROUP BY/HAVING)
4. JOIN — menggabungkan tabel
5. Subquery & CTE
6. Window functions ⭐
7. Feature engineering di SQL
8. SQL → Pandas
9. Jebakan & tips
10. Cheat sheet & latihan

> Contoh memakai sintaks **PostgreSQL** (paling umum di data/ML). Konsep berlaku untuk MySQL, BigQuery, Snowflake, SQLite dengan perbedaan kecil.

---

## 1. Kenapa ML Engineer Butuh SQL

```
[ Database / Data Warehouse ]   ←  data mentah hidup di sini
            │  SQL (ambil + gabung + agregasi + fitur)
            ▼
[ DataFrame Pandas ]            ←  Modul 02
            │  preprocessing + model
            ▼
[ Model ML ]
```

- Data perusahaan tersimpan di tabel relasional (transaksi, user, produk, log).
- **Lebih efisien memproses & memfilter di database** daripada menarik jutaan baris ke Python lalu memproses.
- Banyak feature engineering (agregasi, lag, rolling) bisa & sebaiknya dilakukan di SQL.
- Hampir setiap interview ML/Data ada soal SQL.

---

## 2. Dasar Query

Bayangkan tabel `transaksi`:

| id | user_id | tanggal | jumlah | kategori | status |
|----|---------|---------|--------|----------|--------|

```sql
-- Ambil kolom tertentu
SELECT user_id, jumlah, tanggal
FROM transaksi;

-- Filter baris (WHERE)
SELECT *
FROM transaksi
WHERE jumlah > 100000
  AND status = 'sukses'
  AND tanggal >= '2026-01-01';

-- Urutkan & batasi
SELECT user_id, jumlah
FROM transaksi
ORDER BY jumlah DESC
LIMIT 10;

-- Nilai unik & filter NULL
SELECT DISTINCT kategori FROM transaksi;
SELECT * FROM transaksi WHERE kategori IS NOT NULL;

-- Pola teks & rentang
SELECT * FROM transaksi
WHERE kategori LIKE 'elektro%'          -- diawali 'elektro'
  AND jumlah BETWEEN 50000 AND 200000
  AND status IN ('sukses', 'pending');
```

**Urutan eksekusi logis (penting dipahami):**
`FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT`. Inilah kenapa alias di SELECT belum bisa dipakai di WHERE.

---

## 3. Agregasi (GROUP BY / HAVING)

Inti EDA & feature engineering di SQL — meringkas data per grup.

```sql
-- Statistik per user (ini sudah jadi FITUR!)
SELECT
    user_id,
    COUNT(*)        AS jumlah_transaksi,
    SUM(jumlah)     AS total_belanja,
    AVG(jumlah)     AS rata_belanja,
    MAX(jumlah)     AS belanja_terbesar,
    MIN(tanggal)    AS transaksi_pertama,
    COUNT(DISTINCT kategori) AS variasi_kategori
FROM transaksi
WHERE status = 'sukses'
GROUP BY user_id;

-- HAVING = filter SETELAH agregasi (WHERE tak bisa pakai hasil agregat)
SELECT user_id, SUM(jumlah) AS total
FROM transaksi
GROUP BY user_id
HAVING SUM(jumlah) > 1000000;        -- hanya user "high value"
```

> 🔑 **WHERE** memfilter baris *sebelum* dikelompokkan; **HAVING** memfilter grup *sesudah* agregasi.

### CASE WHEN — logika kondisional (untuk encoding/binning di SQL)
```sql
SELECT
    user_id,
    SUM(CASE WHEN status = 'gagal' THEN 1 ELSE 0 END) AS jumlah_gagal,
    AVG(CASE WHEN kategori = 'elektronik' THEN jumlah END) AS avg_elektronik,
    CASE
        WHEN SUM(jumlah) > 1000000 THEN 'premium'
        WHEN SUM(jumlah) > 100000  THEN 'reguler'
        ELSE 'baru'
    END AS segmen
FROM transaksi
GROUP BY user_id;
```

---

## 4. JOIN — Menggabungkan Tabel

Fitur sering tersebar di banyak tabel. Misal `transaksi`, `users`, `produk`. JOIN menyatukannya.

```sql
SELECT
    t.id, t.jumlah,
    u.nama, u.kota, u.umur,        -- fitur dari tabel users
    p.nama_produk, p.harga         -- fitur dari tabel produk
FROM transaksi t
JOIN users   u ON t.user_id    = u.id
JOIN produk  p ON t.produk_id  = p.id;
```

| Tipe JOIN | Hasil |
|---|---|
| **INNER JOIN** | hanya baris yang cocok di kedua tabel |
| **LEFT JOIN** | semua baris kiri + yang cocok dari kanan (NULL jika tak ada) |
| **RIGHT JOIN** | kebalikan LEFT |
| **FULL OUTER** | semua baris dari kedua tabel |

> Untuk ML, **LEFT JOIN** sangat sering dipakai: "ambil semua transaksi, tempel info user *kalau ada*" — agar tidak kehilangan baris saat data referensi tak lengkap.

```sql
-- LEFT JOIN: tetap simpan transaksi meski user-nya tak ada di tabel users
SELECT t.*, u.kota
FROM transaksi t
LEFT JOIN users u ON t.user_id = u.id;
-- u.kota akan NULL untuk user yang hilang -> bisa jadi sinyal (fitur is_missing)
```

---

## 5. Subquery & CTE

### CTE (Common Table Expression) — `WITH`
Cara rapi memecah query kompleks jadi langkah-langkah terbaca. **Lebih disukai daripada subquery bersarang.**

```sql
WITH belanja_user AS (
    SELECT user_id, SUM(jumlah) AS total, COUNT(*) AS n_trx
    FROM transaksi
    WHERE status = 'sukses'
    GROUP BY user_id
),
rata_global AS (
    SELECT AVG(total) AS avg_total FROM belanja_user
)
SELECT b.user_id, b.total, b.n_trx,
       b.total - r.avg_total AS selisih_dari_rata   -- fitur: deviasi dari rata-rata
FROM belanja_user b
CROSS JOIN rata_global r;
```

---

## 6. Window Functions ⭐ (paling penting untuk ML)

Window function menghitung agregat **tanpa menciutkan baris** — tiap baris tetap ada tapi dapat nilai berbasis "jendela" baris terkait. Ini kunci membuat **fitur per-grup, ranking, lag, dan rolling** — termasuk untuk **time series** ([Modul 11](../11-time-series/README.md)).

Pola umum: `FUNGSI() OVER (PARTITION BY ... ORDER BY ...)`

```sql
SELECT
    user_id, tanggal, jumlah,

    -- Agregat per user TANPA group by (tiap baris tetap tampil)
    SUM(jumlah)  OVER (PARTITION BY user_id) AS total_user,
    AVG(jumlah)  OVER (PARTITION BY user_id) AS avg_user,

    -- Ranking transaksi dalam tiap user
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY tanggal) AS urutan_trx,
    RANK()       OVER (PARTITION BY user_id ORDER BY jumlah DESC) AS rank_jumlah,

    -- LAG/LEAD: nilai baris sebelum/sesudah -> FITUR TIME SERIES
    LAG(jumlah, 1)  OVER (PARTITION BY user_id ORDER BY tanggal) AS jumlah_sebelumnya,
    tanggal - LAG(tanggal) OVER (PARTITION BY user_id ORDER BY tanggal) AS hari_sejak_trx_lalu
FROM transaksi;
```

### Rolling / moving window (rata-rata bergerak) ⭐
```sql
SELECT
    user_id, tanggal, jumlah,
    -- rata-rata 3 transaksi terakhir (termasuk baris ini)
    AVG(jumlah) OVER (
        PARTITION BY user_id ORDER BY tanggal
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_3,

    -- total kumulatif (running total)
    SUM(jumlah) OVER (
        PARTITION BY user_id ORDER BY tanggal
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS kumulatif
FROM transaksi;
```

> 🚨 **Anti-leakage di time series:** untuk fitur prediksi, frame `... AND CURRENT ROW` menyertakan baris saat ini. Kalau target-mu adalah baris ini, gunakan `ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING` agar **hanya memakai masa lalu** — analog `.shift(1)` di Pandas (lihat [Modul 11](../11-time-series/README.md)).

---

## 7. Feature Engineering Langsung di SQL

Contoh nyata: membangun tabel fitur per user untuk model churn — semua di SQL, siap ditarik ke model.

```sql
WITH fitur AS (
    SELECT
        user_id,
        COUNT(*)                                  AS total_trx,
        SUM(jumlah)                               AS total_belanja,
        AVG(jumlah)                               AS avg_belanja,
        COUNT(DISTINCT kategori)                  AS variasi_kategori,
        SUM(CASE WHEN status='gagal' THEN 1 ELSE 0 END)::float
            / COUNT(*)                            AS rasio_gagal,
        MAX(tanggal)                              AS trx_terakhir,
        CURRENT_DATE - MAX(tanggal)               AS hari_sejak_terakhir   -- recency
    FROM transaksi
    GROUP BY user_id
)
SELECT
    f.*,
    u.umur, u.kota,
    CASE WHEN f.hari_sejak_terakhir > 90 THEN 1 ELSE 0 END AS label_churn
FROM fitur f
LEFT JOIN users u ON f.user_id = u.id;
```

Hasil query ini = tabel fitur (satu baris per user) yang langsung bisa `pd.read_sql` → train model.

> 💡 **Prinsip:** lakukan agregasi & filter berat di SQL (dekat data, cepat), lalu tarik hasil yang sudah ramping ke Pandas untuk transformasi akhir & modeling.

---

## 8. SQL → Pandas

```python
import pandas as pd
from sqlalchemy import create_engine

# Koneksi (PostgreSQL contoh; SQLite/MySQL/BigQuery serupa)
engine = create_engine("postgresql://user:pass@host:5432/dbname")

query = """
    SELECT user_id, SUM(jumlah) AS total, COUNT(*) AS n_trx
    FROM transaksi
    WHERE status = 'sukses'
    GROUP BY user_id
"""
df = pd.read_sql(query, engine)     # langsung jadi DataFrame -> lanjut Modul 02

# Untuk eksplorasi cepat tanpa server, SQLite bawaan Python:
import sqlite3
con = sqlite3.connect("data.db")
df = pd.read_sql("SELECT * FROM transaksi LIMIT 5", con)
```

### Kapan proses di SQL vs Pandas?

| Lakukan di **SQL** | Lakukan di **Pandas** |
|---|---|
| Filter & agregasi data besar | Transformasi kompleks / kustom |
| JOIN antar tabel besar | Scaling, encoding untuk model (Modul 02) |
| Mengurangi volume sebelum ditarik | Visualisasi & eksplorasi interaktif |
| Window function & fitur agregat | Operasi yang butuh library Python |

> Aturan praktis: **kecilkan dulu di SQL** (jangan `SELECT *` jutaan baris ke memori), baru poles di Pandas.

---

## 9. Jebakan & Tips

1. **`SELECT *` pada tabel besar** → boros memori & lambat. Ambil hanya kolom yang perlu.
2. **JOIN tanpa kondisi / kunci salah** → ledakan baris (cartesian product). Cek `COUNT(*)` sebelum & sesudah JOIN.
3. **NULL itu licik** → `NULL = NULL` bukan TRUE; pakai `IS NULL`. Agregat seperti `AVG` mengabaikan NULL.
4. **Leakage waktu di window** → jangan masukkan baris/masa depan saat membuat fitur prediksi (bagian 6).
5. **Lupa filter status/duplikat** → fitur jadi salah. Selalu cek kualitas data dulu (EDA via SQL).
6. **Tipe data** → hati-hati pembagian integer (`1/2 = 0` di beberapa DB); cast ke float: `SUM(x)::float / COUNT(*)`.

---

## 10. Cheat Sheet & Latihan

```sql
SELECT kolom, AGG(...) OVER (PARTITION BY ... ORDER BY ...)   -- window
FROM   tabel_a a
JOIN   tabel_b b ON a.id = b.a_id          -- gabung tabel
WHERE  kondisi_baris                       -- filter sebelum grup
GROUP  BY kolom
HAVING kondisi_grup                        -- filter setelah grup
ORDER  BY kolom DESC
LIMIT  n;
-- Pecah query kompleks dengan WITH cte AS (...)
```

### Latihan
1. Dari tabel transaksi, hitung total & rata-rata belanja per user, lalu filter user dengan >5 transaksi (GROUP BY + HAVING).
2. JOIN transaksi dengan users; buat fitur: total belanja, kota, umur. Pakai LEFT JOIN dan jelaskan kenapa.
3. Pakai window function untuk membuat fitur **recency** (`hari_sejak_transaksi_terakhir`) dan **rolling average 3 transaksi** per user.
4. Buat fitur time series anti-leakage: rata-rata 7 hari terakhir **tidak termasuk hari ini** (`ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING`).
5. Tulis satu CTE yang menghasilkan tabel fitur lengkap untuk model churn (lihat bagian 7), lalu tarik ke Pandas dengan `pd.read_sql` dan latih model sederhana.
6. (Latihan gratis) Kerjakan soal di [sqlbolt.com](https://sqlbolt.com), [pgexercises.com](https://pgexercises.com), atau StrataScratch (soal SQL gaya interview data).

⬅️ Kembali ke [Daftar Modul](../README.md) · Terkait: [Modul 02 · Data & Feature Engineering](../02-data-feature-engineering/README.md), [Modul 11 · Time Series](../11-time-series/README.md)
