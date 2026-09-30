# Belajar Python dan ML: API Prediksi Harga Rumah

Proyek pemula untuk mempelajari Python, REST API, dan machine learning melalui satu alur kecil:

**Buat data → latih model → evaluasi → simpan model → prediksi melalui API.**

Kita memakai Python 3.12, FastAPI, dan `LinearRegression` dari scikit-learn. Tidak perlu API key, database, atau layanan berbayar. Data dibuat secara sintetis; hasilnya hanya untuk latihan, bukan estimasi harga pasar nyata.

## 1. Siapkan Python

Buka terminal di folder proyek ini. Perintah panduan ini menggunakan terminal macOS/Linux.

Jika Python 3.12 belum terpasang, di macOS dengan Homebrew jalankan:

```sh
brew install python@3.12
python3.12 --version
```

Alternatif: unduh installer Python 3.12 untuk sistem operasimu dari [situs Python](https://www.python.org/downloads/). Bila Homebrew sudah memasang Python tetapi perintah tidak ditemukan, gunakan `"$(brew --prefix python@3.12)/bin/python3.12"` sebagai pengganti `python3.12`.

Buat lingkungan baru agar paket proyek terpisah dari Python sistem. Folder `.venv` lama tidak perlu dihapus:

```sh
python3.12 -m venv .venv312
source .venv312/bin/activate
python --version
python -m pip install -r requirements.txt
```

Pastikan versi yang tampil adalah **3.12.x**. Jika `.venv312` sudah disiapkan, cukup aktifkan dan instal dependensinya. Versi paket dikunci di `requirements.txt` untuk membantu mengulang lingkungan latihan.

## 2. Latih model

```sh
python train.py
```

Program membuat 500 baris data di memori, membaginya menjadi 400 data training dan 100 data pengujian, kemudian menyimpan model di `artifacts/house_model.joblib`. Tidak ada unduhan dataset. Seed 42 membuat data serta pembagian train/test dapat diulang.

Tiga **fitur** (masukan) yang digunakan:

| Nama | Arti | Rentang |
| --- | --- | --- |
| `luas_m2` | Luas bangunan dalam meter persegi | 20–500 |
| `jumlah_kamar` | Jumlah kamar, bilangan bulat | 1–10 |
| `usia_tahun` | Usia bangunan dalam tahun | 0–50 |

**Target** adalah harga rumah dalam rupiah. Harga contoh dibuat dengan rumus:

```text
200 juta + luas × 8 juta + kamar × 40 juta − usia × 2 juta + variasi acak
```

Variasi acak mengikuti distribusi normal dengan simpangan baku Rp25 juta. Rumus ini sengaja sederhana. Model tidak diberi rumus tersebut: `fit()` mempelajari koefisien dari pasangan fitur dan harga pada data training.

Program menampilkan **MAE (Mean Absolute Error)** model serta pembanding yang selalu menebak rata-rata harga training. MAE Rp20 juta berarti selisih absolut prediksi dan target rata-rata sekitar Rp20 juta pada data pengujian; semakin kecil semakin baik. Ini bukan persentase akurasi atau jaminan setiap prediksi meleset sebesar itu.

Data pengujian tidak digunakan untuk melatih model. Karena dataset sintetis dibuat dengan hubungan linear, regresi linear cocok dan hasilnya biasanya bagus. Itu tidak membuktikan akurasi pada pasar rumah nyata yang juga dipengaruhi lokasi, kondisi bangunan, dan faktor lain.

## 3. Jalankan REST API

```sh
python -m uvicorn main:app --reload
```

Biarkan terminal ini berjalan. Buka [dokumentasi interaktif](http://127.0.0.1:8000/docs), pilih **POST /predict → Try it out**, masukkan contoh berikut, lalu klik **Execute**:

```json
{
  "luas_m2": 100,
  "jumlah_kamar": 3,
  "usia_tahun": 5
}
```

Respons berisi satu angka `prediksi_harga_rupiah`, sekitar Rp1,1 miliar untuk contoh tersebut. Angka tepatnya berasal dari model hasil training.

API memuat model sekali saat mulai. Setelah training ulang, hentikan API dengan **Ctrl+C** lalu jalankan lagi agar model terbaru dimuat. Folder `/` tidak mempunyai halaman; gunakan `/docs` atau `/health`.

### Mencoba dari terminal lain

```sh
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"luas_m2":100,"jumlah_kamar":3,"usia_tahun":5}'
```

| Endpoint | Hasil |
| --- | --- |
| `GET /health` | 200 dengan `{"status":"ok","model_ready":true}` jika siap |
| `POST /predict` | 200 dengan `prediksi_harga_rupiah` |
| `GET /docs` | Dokumentasi dan formulir percobaan API |

Masukan harus berupa angka JSON, bukan teks seperti `"100"`. Luas dan usia boleh desimal; kamar harus bilangan bulat. Kolom tambahan, kolom hilang, tipe salah, dan angka di luar rentang menghasilkan **422** dengan penjelasan validasi.

Jika model belum dibuat, `/health` dan prediksi valid menghasilkan **503** dengan instruksi `python train.py`. `/docs` tetap bisa dibuka. Jalankan training lalu restart API. Jika file model rusak, startup gagal: latih ulang untuk mengganti file tersebut. Hanya muat file `joblib` hasil training sendiri.

## 4. Pahami kode Python

Baca file dengan urutan `train.py` → `model.py` → `main.py`, lalu lihat contoh uji dalam `tests/test_project.py`.

| Konsep | Contoh dalam proyek |
| --- | --- |
| Variabel | `features` menyimpan tabel masukan dan `target` menyimpan harga |
| Fungsi | `generate_data()` mengelompokkan langkah pembuatan data; `return` mengembalikan hasil |
| Tipe data | `int` untuk kamar, `float` untuk angka desimal, `str` untuk pesan |
| List | `[[luas_m2, jumlah_kamar, usia_tahun]]` adalah satu baris fitur untuk prediksi |
| Dictionary | `metrics` berisi pasangan nama metrik dan nilainya |
| Impor modul | `from model import save_model` memakai fungsi dari file lain |
| Class | `HouseInput` mendefinisikan bentuk dan aturan data permintaan |
| Type hint | `luas_m2: float` menjelaskan tipe; Pydantic memakai tipe dan `Field` untuk validasi API |
| Kondisi | `if app.state.model is None` menangani model yang belum tersedia |
| Exception | `try/except FileNotFoundError` menangani file yang belum ada |
| Entry point | `if __name__ == "__main__"` menjalankan training hanya saat file dieksekusi langsung |

**JSON** adalah format pertukaran data, menyerupai dictionary Python, tetapi nama kolom memakai tanda kutip ganda. **REST API** menyediakan alamat yang bisa dipanggil program lain: GET membaca status, POST mengirim data untuk diproses. Kode HTTP 200 berarti berhasil, 422 berarti masukan tidak valid, 503 berarti layanan prediksi belum siap.

**Training** (`model.fit`) mempelajari pola dari contoh berlabel. **Prediksi** (`model.predict`) memakai pola yang sudah dipelajari untuk masukan baru, tanpa melatih ulang setiap permintaan. ML adalah salah satu pendekatan dalam AI; proyek ini tidak menggunakan chatbot atau model bahasa.

Dekorator `@app.post` menghubungkan fungsi Python dengan endpoint. `lifespan` menjalankan pemuatan model saat server mulai. Fungsi endpoint dibuat biasa (`def`); kamu tidak perlu mempelajari pemrograman asynchronous dahulu untuk mencoba proyek ini.

## 5. Jalankan pengujian

Dalam lingkungan `.venv312` yang aktif:

```sh
python -m pytest -q
```

Pengujian memeriksa kualitas model dibanding tebakan rata-rata, penyimpanan/pemuatan ulang, data yang dapat diulang, prediksi normal serta batas rentang, validasi masukan, dokumentasi, pemuatan sekali, dan model yang belum tersedia. Pengujian menggunakan file sementara dan tidak menimpa model latihanmu.

## 6. Latihan lanjutan

1. Ubah luas pada permintaan sambil mempertahankan kamar dan usia. Amati perubahan prediksi.
2. Ubah jumlah data melalui nilai default `n_samples` pada `generate_data()`, latih ulang, lalu bandingkan MAE. Pengujian jumlah data perlu disesuaikan jika default berubah.
3. Perbesar variasi acak harga dan lihat pengaruhnya pada MAE.
4. Cetak `model.coef_` serta `model.intercept_` setelah training dan bandingkan dengan rumus harga. Urutan koefisien: luas, kamar, usia.
5. Setelah memahami alurnya, coba menambah fitur baru. Sesuaikan generator, urutan fitur prediksi, skema API, pengujian, dan latih ulang.

Untuk keluar dari lingkungan virtual, jalankan `deactivate`. File model dan lingkungan virtual diabaikan Git; setelah menyalin proyek ke komputer baru, instal dependensi dan jalankan training lagi.

Referensi: [FastAPI](https://fastapi.tiangolo.com/tutorial/), [regresi linear scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), dan [MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).
