# 00.0 · Dasar-Dasar Python (dari Nol)

> Sebelum NumPy & Pandas ([00.1](./01-python-data-science.md)), kamu butuh dasar Python yang kokoh. Materi ini untuk yang **benar-benar baru** — kalau kamu sudah lancar variabel, loop, fungsi, list, dan dict, **langsung loncat ke [00.1](./01-python-data-science.md)**.

## Tujuan Belajar
- Menjalankan kode Python & memahami sintaksnya.
- Menguasai tipe data, operator, kondisi, loop, fungsi.
- Menguasai struktur data inti: **list, tuple, dict, set**.
- Memahami error handling, modul, dan dasar OOP.

---

## 1. Menjalankan Python

Tiga cara:
```bash
python script.py        # jalankan file
python                  # mode interaktif (REPL) — bagus untuk coba cepat
jupyter lab             # notebook — ideal untuk belajar & data science
```

Program pertama:
```python
print("Halo, AI/ML!")
```

**Komentar** (diabaikan Python, untuk penjelasan):
```python
# ini komentar satu baris
"""
ini komentar
beberapa baris
"""
```

> ⚠️ Python pakai **indentasi (spasi)** untuk menandai blok kode, bukan kurung kurawal. Indentasi yang salah = error. Standar: **4 spasi**.

---

## 2. Variabel & Tipe Data

Variabel tak perlu dideklarasikan tipenya — Python menyimpulkan otomatis.

```python
nama = "Budi"        # str (teks)
umur = 25            # int (bilangan bulat)
tinggi = 1.75        # float (desimal)
aktif = True         # bool (True/False)
kosong = None        # None (ketiadaan nilai)

type(umur)           # <class 'int'>  -> cek tipe
```

### Konversi tipe (casting)
```python
int("10")            # 10
float("3.14")        # 3.14
str(100)             # "100"
int(3.9)             # 3  (dipotong, bukan dibulatkan)
```

---

## 3. Operator

```python
# Aritmetika
10 + 3      # 13
10 - 3      # 7
10 * 3      # 30
10 / 3      # 3.333...  (selalu float)
10 // 3     # 3   (pembagian bulat / floor)
10 % 3      # 1   (sisa bagi / modulo)
10 ** 3     # 1000 (pangkat)

# Perbandingan -> menghasilkan bool
5 > 3       # True
5 == 5      # True   (== untuk membandingkan, = untuk assign)
5 != 4      # True

# Logika
True and False   # False
True or False    # True
not True         # False
```

---

## 4. String (Teks)

```python
s = "Machine Learning"

len(s)              # 16  (panjang)
s.upper()           # "MACHINE LEARNING"
s.lower()           # "machine learning"
s.replace("a", "@") # "M@chine Le@rning"
s.split(" ")        # ['Machine', 'Learning']
"  spasi  ".strip() # "spasi"  (hapus spasi tepi)

# Indexing & slicing (mulai dari 0)
s[0]                # 'M'
s[-1]               # 'g'  (terakhir)
s[0:7]              # 'Machine'  (indeks 0 sampai 6)

# f-string -> cara terbaik menyisipkan variabel
nama, skor = "Ani", 95.5
print(f"{nama} mendapat nilai {skor}")          # Ani mendapat nilai 95.5
print(f"Pembulatan: {skor:.0f}")                # Pembulatan: 96
```

---

## 5. Input & Output

```python
nama = input("Siapa namamu? ")     # baca input user (selalu str)
umur = int(input("Umurmu? "))      # konversi ke angka
print(f"Halo {nama}, umurmu {umur}")
```

---

## 6. Kondisi (if / elif / else)

```python
nilai = 75

if nilai >= 90:
    grade = "A"
elif nilai >= 75:
    grade = "B"
elif nilai >= 60:
    grade = "C"
else:
    grade = "D"

print(grade)        # B

# Bentuk ringkas (ternary)
status = "lulus" if nilai >= 60 else "gagal"
```

---

## 7. Loop (Perulangan)

### for — mengulang sejumlah/atas koleksi
```python
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):       # 1, 2, 3, 4, 5
    print(i)

buah = ["apel", "mangga", "jeruk"]
for b in buah:
    print(b)

for i, b in enumerate(buah):    # dapat indeks + nilai
    print(i, b)
```

### while — selama kondisi benar
```python
n = 5
while n > 0:
    print(n)
    n -= 1          # n = n - 1

# Kontrol loop
for i in range(10):
    if i == 3:
        continue    # lewati iterasi ini
    if i == 7:
        break       # hentikan loop
    print(i)
```

---

## 8. Struktur Data Inti

### List — koleksi terurut, bisa diubah ⭐ (paling sering dipakai)
```python
angka = [10, 20, 30, 40]

angka[0]            # 10
angka[-1]           # 40
angka[1:3]          # [20, 30]  (slicing)

angka.append(50)    # tambah di akhir -> [10,20,30,40,50]
angka.insert(0, 5)  # sisip di indeks 0
angka.remove(20)    # hapus nilai 20
angka.pop()         # buang & kembalikan elemen terakhir
len(angka)          # panjang
30 in angka         # True  (cek keanggotaan)
angka.sort()        # urutkan (mengubah list asli)

campur = [1, "dua", 3.0, True]   # boleh beda tipe
matriks = [[1, 2], [3, 4]]       # list di dalam list (nested)
```

### Tuple — seperti list, tapi TAK bisa diubah (immutable)
```python
titik = (3, 5)
titik[0]            # 3
# titik[0] = 9      # ❌ Error! tuple tak bisa diubah
x, y = titik        # unpacking -> x=3, y=5
```
Dipakai saat data tak boleh berubah (mis. koordinat, mengembalikan banyak nilai dari fungsi).

### Dictionary — pasangan key:value ⭐ (sangat penting)
```python
orang = {
    "nama": "Sari",
    "umur": 30,
    "kota": "Bandung",
}

orang["nama"]               # "Sari"
orang["email"] = "s@x.com"  # tambah/ubah
orang.get("hp", "tidak ada") # ambil aman dengan default
del orang["kota"]           # hapus key

orang.keys()                # semua key
orang.values()              # semua value
orang.items()               # pasangan (key, value)

for key, value in orang.items():
    print(key, "->", value)
```

### Set — koleksi nilai unik, tak terurut
```python
s = {1, 2, 2, 3, 3, 3}      # -> {1, 2, 3}  (duplikat otomatis hilang)
s.add(4)
{1, 2, 3} & {2, 3, 4}       # irisan -> {2, 3}
{1, 2, 3} | {3, 4}          # gabungan -> {1, 2, 3, 4}
```

| Struktur | Terurut? | Bisa diubah? | Duplikat? | Untuk |
|---|---|---|---|---|
| **list** | ya | ya | ya | urutan data, koleksi umum |
| **tuple** | ya | tidak | ya | data tetap, return ganda |
| **dict** | ya* | ya | key unik | data berlabel (key→value) |
| **set** | tidak | ya | tidak | nilai unik, operasi himpunan |

---

## 9. Fungsi

Membungkus kode agar bisa dipakai ulang.

```python
def luas_persegi(sisi):
    return sisi * sisi

luas_persegi(5)         # 25

# Parameter default
def sapa(nama, salam="Halo"):
    return f"{salam}, {nama}!"

sapa("Budi")                    # "Halo, Budi!"
sapa("Budi", "Selamat pagi")    # "Selamat pagi, Budi!"

# Mengembalikan banyak nilai (sebagai tuple)
def statistik(angka):
    return min(angka), max(angka), sum(angka) / len(angka)

lo, hi, rata = statistik([3, 7, 2, 9])

# lambda — fungsi singkat satu baris
kuadrat = lambda x: x ** 2
kuadrat(4)              # 16
```

> Fungsi adalah pondasi kode bersih. Kalau kamu menyalin-tempel kode yang sama, jadikan fungsi.

---

## 10. List Comprehension (idiom Python penting)

Cara ringkas & cepat membuat list — sangat sering muncul di kode ML.

```python
# Cara panjang:
kuadrat = []
for x in range(10):
    kuadrat.append(x ** 2)

# Cara Pythonic:
kuadrat = [x ** 2 for x in range(10)]

# Dengan kondisi:
genap = [x for x in range(20) if x % 2 == 0]

# Dict comprehension:
peta = {x: x ** 2 for x in range(5)}    # {0:0, 1:1, 2:4, 3:9, 4:16}
```

---

## 11. Error Handling

```python
try:
    hasil = 10 / 0
except ZeroDivisionError:
    print("Tidak bisa bagi nol!")
except Exception as e:
    print("Error lain:", e)
finally:
    print("Selalu dijalankan")
```
Penting saat membaca file/data yang mungkin gagal.

---

## 12. Modul & Import

Python punya pustaka standar luas + jutaan paket pihak ketiga (inilah kenapa ia favorit AI/ML).

```python
import math
math.sqrt(16)               # 4.0

from random import randint
randint(1, 6)               # angka acak 1-6

import numpy as np          # konvensi alias (Modul 00.1)
```

Install paket dengan pip:
```bash
pip install numpy pandas
```

---

## 13. Dasar OOP (Class)

Kamu tak harus jadi ahli OOP untuk ML, tapi perlu **membacanya** — model PyTorch & scikit-learn berbasis class.

```python
class Mahasiswa:
    def __init__(self, nama, nilai):    # konstruktor: dipanggil saat objek dibuat
        self.nama = nama                # atribut
        self.nilai = nilai

    def lulus(self):                    # method
        return self.nilai >= 60

m = Mahasiswa("Andi", 75)
m.nama          # "Andi"
m.lulus()       # True
```
Pola `class ... def __init__(self) ... def forward(self)` inilah yang akan kamu lihat saat membuat neural network di [Modul 04](../04-deep-learning/README.md).

---

## Latihan
1. Buat program yang menerima nilai 0–100 dari input dan mencetak grade (A/B/C/D) memakai if/elif.
2. Tulis fungsi `hitung_rata(angka)` yang menerima list dan mengembalikan rata-rata. Tangani kasus list kosong dengan error handling.
3. Diberi list `[5, 3, 8, 1, 9, 2]`, pakai list comprehension untuk membuat list baru berisi hanya angka > 4, masing-masing dikuadratkan.
4. Buat dict berisi data 3 orang (nama, umur). Loop dan cetak siapa yang berumur > 25.
5. Buat class `RekeningBank` dengan method `setor()`, `tarik()`, dan atribut `saldo`. Cegah penarikan melebihi saldo.
6. Tulis program yang menghitung berapa kali tiap kata muncul dalam sebuah kalimat (petunjuk: `split()` + dict).

> 💡 **Cara belajar tercepat:** ketik ulang setiap contoh, ubah-ubah nilainya, dan lihat apa yang terjadi. Jangan cuma membaca.

➡️ Lanjut: [00.1 · Python untuk Data Science (NumPy & Pandas)](./01-python-data-science.md)
