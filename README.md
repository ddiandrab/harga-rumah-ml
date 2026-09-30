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

Program membuat 1.000 baris data di memori, membaginya menjadi 800 data training dan 200 data pengujian, kemudian menyimpan model di `artifacts/house_model.joblib`. Tidak ada unduhan dataset. Seed 42 membuat data serta pembagian train/test dapat diulang.

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

API memuat model sekali saat mulai. Setelah training ulang, hentikan API dengan **Ctrl+C** lalu jalankan lagi agar model terbaru dimuat. Folder `/` sekarang menampilkan halaman RumahLab; `/docs` tetap tersedia untuk mencoba API.

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

Jika model belum dibuat, `/health` dan prediksi valid menghasilkan **503** dengan instruksi `python train.py`. `/docs` tetap bisa dibuka. Jalankan training lalu restart API. Jika file model rusak, mode tersebut tidak tersedia: latih ulang untuk mengganti file tersebut. Hanya muat file `joblib` hasil training sendiri.

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

---

# Tahap lanjutan: rumah Jabodetabek dengan data nyata

## Mulai cepat

Aktifkan `.venv312`, kemudian perbarui dependensi (versi lanjut menambahkan pandas):

```sh
source .venv312/bin/activate
python -m pip install -r requirements.txt
python train.py
```

Latihan sintetis sekarang memakai **1.000 baris**, dengan 800 untuk training dan 200 untuk pengujian. Endpoint `/predict` tetap menerima format lama. Artefak lama tetap dapat dipakai; training ulang melengkapinya dengan metadata evaluasi.

## Memperoleh dataset asli

1. Buka [Daftar Harga Rumah Jabodetabek — Nafis Barizki di Kaggle](https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek).
2. Baca deskripsi, versi, dan bagian **License** di halaman sumber sebelum menggunakan atau membagikan dataset. Proyek ini tidak memberikan lisensi ulang atas data tersebut. Lihat juga [penjelasan pengumpulan dan pembersihan oleh pembuat dataset](https://nbarizki.github.io/posts/jabodetabekhousepricing/_build/html/README.html).
3. Pilih **Download** (Kaggle mungkin meminta login), ekstrak ZIP, lalu letakkan `jabodetabek_house_price.csv` di folder `data/raw/`. Buat folder jika belum ada: `mkdir -p data/raw`.
4. Jika file sudah ada dari setup proyek ini, kamu dapat langsung melatih model. CSV dan model diabaikan Git, sehingga perlu disiapkan kembali pada komputer lain.

Dataset yang diuji berisi 3.553 baris iklan properti. Ini adalah harga penawaran pada dataset, bukan harga transaksi atau harga pasar hari ini. Jangan menganggap cakupan model mewakili seluruh Indonesia.

```sh
python train.py --dataset jabodetabek --csv data/raw/jabodetabek_house_price.csv
python -m uvicorn main:app --reload
```

Buka [halaman RumahLab](http://127.0.0.1:8000/), pilih **Lanjutan · Data Jabodetabek**, tekan **Isi contoh**, lalu **Prediksi harga**. Bandingkan hasil dengan mengubah satu fitur. Tabel di bawah formulir menampilkan evaluasi. Setelah training ulang, restart server agar model terbaru dimuat.

Tidak ada unduhan atau training saat API dimulai. Jika model belum tersedia atau rusak, halaman tetap terbuka dan menunjukkan perintah training; mode lain tetap dapat digunakan. Model tidak diganti otomatis dengan model sintetis.

## Alur kode dan pembersihan data

Baca `data_processing.py` → `train.py` → `model.py` → `main.py` → `static/app.js`. Halaman memakai HTML/CSS/JavaScript biasa; tidak memerlukan Node.js atau frontend framework.

| Kolom CSV | Masukan API | Arti |
| --- | --- | --- |
| `city` | `kota` | Nama kota sesuai daftar model |
| `land_size_m2` | `luas_tanah_m2` | Luas tanah, m² |
| `building_size_m2` | `luas_bangunan_m2` | Luas bangunan, m² |
| `bedrooms` | `kamar_tidur` | Jumlah kamar tidur |
| `bathrooms` | `kamar_mandi` | Jumlah kamar mandi |
| `price_in_rp` | Tidak dikirim | Target harga dalam rupiah |

Gunakan CSV hasil unduhan yang sudah mempunyai kolom tersebut, bukan hasil scraping mentah. Angka harus berupa angka biasa (contoh `1500000000`), bukan teks `Rp 1,5 miliar`.

`read_houses()` membaca CSV dengan pandas, memeriksa kolom wajib, mengubah angka, dan merapikan spasi/huruf nama kota. Baris dengan data kosong atau tidak terbaca dibuang; luas/harga harus positif dan terhingga, kamar harus positif serta bulat. Duplikat ID iklan (jika tersedia) dan duplikat kombinasi fitur serta harga dihapus sebelum pembagian data. Jumlah yang dibuang dicetak per kategori; tidak ada perubahan pada CSV asli. Minimal 30 baris bersih diperlukan.

Nilai ekstrem yang masih valid tidak otomatis dihapus. Beberapa iklan mungkin tidak akurat; ini bagian penting dalam mempelajari kualitas data. Penghapusan duplikat sederhana juga tidak menjamin semua iklan rumah yang sama dengan ID berbeda sudah terdeteksi.

## Mengapa membandingkan dua model?

- **Dataset** adalah contoh yang dipelajari. **Model** adalah pola/parameter hasil training, bukan file CSV itu sendiri. Model awal kita sudah ML sungguhan; versi ini memakai data nyata.
- **Regresi linear** mempelajari hubungan berbentuk penjumlahan berbobot seperti pada latihan pertama.
- **Random Forest** menggabungkan prediksi 100 pohon keputusan dan dapat mempelajari pola non-linear. Tidak membutuhkan GPU.
- **Kategori dan one-hot encoding:** nama kota diubah menjadi kolom indikator angka agar bisa dibaca model. Encoder dapat menangani kategori baru selama evaluasi; API sengaja menolak kota di luar daftar training.
- **Pipeline** menyatukan encoding, standardisasi angka, dan model. Statistik preprocessing dipelajari hanya dari bagian training sehingga informasi evaluasi tidak bocor ke model. Standardisasi membantu skala fitur regresi linear; Random Forest tidak membutuhkannya, tetapi memakai pipeline sama agar contoh mudah diikuti.

Urutan pembelajaran data nyata:

1. Pisahkan 20% sebagai data pengujian yang disimpan hingga tahap akhir.
2. Bagi sisanya menjadi training dan validasi: total sekitar 60%/20%/20%, dengan pembulatan jumlah baris dan seed 42.
3. Latih kedua model hanya pada training. Ukur MAE dan R² pada validasi; tampilkan juga pembanding yang selalu menebak median harga training.
4. Pilih model dengan MAE validasi terkecil. Latih ulang algoritma terpilih pada gabungan training dan validasi (80%).
5. Evaluasi model final dan pembanding median pada data pengujian (20%). Simpan seluruh metadata bersama estimator.

**Overfitting** terjadi ketika model terlalu menghafal data training dan lemah pada contoh baru. Gunakan data validasi untuk eksperimen; jangan memilih parameter berulang kali berdasarkan data pengujian. Split acak ini adalah latihan awal, bukan bukti kemampuan memprediksi harga masa depan atau wilayah yang belum pernah dilihat.

**MAE** mempunyai satuan rupiah. **R²** bukan persentase akurasi: 1 berarti prediksi sempurna, 0 setara menebak rata-rata target pengujian, dan negatif berarti lebih buruk menurut ukuran kesalahan kuadrat tersebut. MAE dan R² mengukur aspek berbeda, sehingga kesimpulannya bisa berbeda.

## API mode lanjutan

```sh
curl http://127.0.0.1:8000/models

curl -X POST http://127.0.0.1:8000/predict/jabodetabek \
  -H 'Content-Type: application/json' \
  -d '{"kota":"Bekasi","luas_tanah_m2":100,"luas_bangunan_m2":80,"kamar_tidur":3,"kamar_mandi":2}'
```

Respons: `prediksi_harga_rupiah`, nama `model`, serta daftar `warnings`. Angka positif di luar rentang training menghasilkan peringatan; hasilnya kurang andal. Kombinasi yang tidak menghasilkan harga positif/terhingga atau terlalu ekstrem untuk dihitung ditolak dengan 422, bukan diubah diam-diam menjadi nol.

`GET /models` menampilkan status setiap mode, perintah training, waktu training, versi pustaka, daftar kota, rentang fitur, pembersihan, dan evaluasi. Gunakan daftar kota tersebut; Jakarta dibedakan menjadi wilayah seperti Jakarta Selatan dan Jakarta Barat. `GET /health` tetap memeriksa mode sintetis demi kompatibilitas versi awal.

Kolom hilang/tambahan, tipe salah, kota yang tidak dikenal, angka nol/negatif, NaN, atau Infinity ditolak dengan 422. Model belum tersedia menghasilkan 503. Luas boleh desimal; kamar harus bilangan bulat JSON. Masukan baru tidak disimpan di server.

Model final disimpan pada `artifacts/jabodetabek.joblib`; model sintetis pada `artifacts/house_model.joblib`. Setiap artefak baru berisi estimator dan metadata sehingga urutan fitur serta hasil evaluasinya tetap bersama. Muat hanya artefak hasil training sendiri.

## Pengujian dan latihan lanjutan

```sh
python -m pytest -q
```

Tes menggunakan CSV kecil buatan di folder sementara, bukan unduhan internet. Cakupannya meliputi pembersihan, duplikasi, kolom hilang, training, pemilihan model, penyimpanan ulang, API lama/baru, masukan tidak valid, peringatan rentang, halaman web, serta model hilang/rusak.

Coba ubah `max_depth` atau `min_samples_leaf` pada Random Forest dan bandingkan **hasil validasi**. Jelaskan alasan pilihan sebelum melihat data pengujian. Untuk eksperimen berikutnya, pelajari kualitas iklan dan fitur lokasi lebih detail; tidak perlu langsung menambah algoritma yang lebih rumit.

## Hasil verifikasi data nyata

Pada pengujian lokal Python 3.12, CSV unduhan Kaggle berisi 3.553 baris. Setelah membuang 39 baris kosong/tidak terbaca dan 915 duplikat, tersisa 2.599 baris. Random Forest dipilih berdasarkan MAE validasi sekitar Rp2,23 miliar (regresi linear Rp2,93 miliar).

Pada 520 baris pengujian, MAE model final sekitar **Rp2,60 miliar**, dibanding median **Rp4,45 miliar**. R² model **−0,0438**: meskipun MAE lebih baik dari pembanding median, kemampuan model menurut kesalahan kuadrat masih lemah. Jangan menyamakan “berhasil dilatih” dengan “akurat untuk digunakan”. Tidak dilakukan tuning berdasarkan hasil test ini. Nilai bisa berubah jika versi CSV atau pustaka berubah; metadata training adalah sumber angka untuk run terbaru.
