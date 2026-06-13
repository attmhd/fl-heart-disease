# Perbedaan Evaluasi Centralized vs FedAvg (Distributed) pada Metrik MLflow

Dokumen ini menjelaskan perbedaan mendasar antara metrik evaluasi **Centralized (Terpusat)** dan **FedAvg / Distributed (Terdistribusi)** yang dicatat pada dashboard MLflow dalam arsitektur Federated Learning ini.

---

## 1. Centralized Evaluation (Evaluasi Terpusat)

Metrik dengan prefix **`centralized_`** (seperti `centralized_loss` dan `centralized_accuracy`) merepresentasikan evaluasi model global langsung di sisi server menggunakan dataset uji tunggal yang terpusat.

* **Lokasi Eksekusi:** Di jalankan sepenuhnya di sisi **Server**.
* **Dataset yang Digunakan:** Dataset uji global (di server) yang tidak pernah dibagikan atau dilihat oleh client (ditentukan oleh variabel `SERVER_DATASET_PATH`).
* **Mekanisme:**
  1. Setelah server selesai mengagregasi bobot model dari para client pada suatu ronde, server memuat bobot hasil agregasi tersebut ke dalam instance model global baru.
  2. Server melakukan pengujian (`model.predict` dan `model.score`) menggunakan dataset uji global tersebut.
  3. Hasil evaluasi langsung dicatat ke MLflow.
* **Tujuan utama:** Mengetahui performa model global yang sebenarnya terhadap satu set data pengujian standar yang konsisten dan independen.

---

## 2. FedAvg / Distributed Metrics (Metrik Terdistribusi)

Metrik dengan prefix **`fedavg_`** (seperti `fedavg_train_accuracy`, `fedavg_val_accuracy`, `fedavg_eval_accuracy`) merepresentasikan rata-rata performa model dari hasil evaluasi lokal masing-masing client.

* **Lokasi Eksekusi:** Dijalankan di sisi masing-masing **Client**.
* **Dataset yang Digunakan:** Dataset lokal milik client sendiri (`X_train`, `y_train`, `X_test`, `y_test` lokal) yang tidak pernah dikirim ke server demi menjaga privasi.
* **Mekanisme:**
  1. Setiap client menerima bobot model global dari server.
  2. Client melakukan pelatihan (*local fit*) atau evaluasi (*local evaluate*) menggunakan data lokal mereka sendiri.
  3. Client menghitung metrik lokal mereka (misal: akurasi dan loss lokal) lalu mengirimkan metrik tersebut beserta jumlah sampel data mereka ke server.
  4. Server menerima laporan metrik dari semua client dan menggabungkannya menggunakan **rata-rata tertimbang** (*weighted average*) berdasarkan jumlah sampel data masing-masing client:
     $$\text{FedAvg Metric} = \frac{\sum (Metric_{client} \times N_{client})}{\sum N_{client}}$$
  5. Hasil rata-rata tertimbang tersebut dicatat ke MLflow.
* **Tujuan utama:** Mengetahui bagaimana performa model global jika diuji pada kumpulan data lokal client secara kolektif tanpa melanggar privasi data.

---

## Ringkasan Perbandingan

| Dimensi | Centralized Evaluation | FedAvg / Distributed Metrics |
| :--- | :--- | :--- |
| **Tempat Evaluasi** | Server | Client |
| **Dataset** | Dataset Uji Tunggal (di Server) | Dataset Lokal Masing-masing Client |
| **Perhitungan Metrik** | Evaluasi langsung pada satu data uji | Rata-rata tertimbang (*weighted average*) metrik client |
| **Privasi Data** | Data uji harus diunggah ke server | Data tetap di client (hanya metrik ringkasan yang dikirim) |
| **Prefix MLflow** | `centralized_...` | `fedavg_...` |

---

## 3. Di Mana Proses Agregasi FedAvg Berjalan?

Dalam arsitektur Flower, algoritma **FedAvg (Federated Averaging)** berjalan di dalam kelas **`MLflowStrategy`** (yang mewarisi kelas `fl.server.strategy.FedAvg` dari Flower) pada berkas [server_utils.py](file:///home/attmhd/Dokumen/Kuliah/TA/heart-disease-fl/src/utils/server_utils.py) melalui dua metode utama:

### A. Agregasi Parameter Model (`aggregate_fit`)
Metode ini dijalankan di server setelah client menyelesaikan training lokal:
* **Fungsi:** `MLflowStrategy.aggregate_fit(self, rnd, results, failures)`
* **Proses Agregasi:**
  1. Menerima parameter lokal baru ($w_i$) dan jumlah sampel data ($n_i$) dari tiap client yang berhasil menyelesaikan training (`results`).
  2. Memanggil `super().aggregate_fit(rnd, results, failures)`, yang di dalamnya memanggil fungsi utilitas Flower untuk menghitung rata-rata parameter tertimbang:
     $$w_{new} = \sum \left( \frac{n_i}{N} \times w_i \right)$$
     di mana $N$ adalah total sampel dari seluruh client pada ronde tersebut.
  3. Menyimpan parameter global baru hasil agregasi tersebut sebagai basis ronde berikutnya.

### B. Agregasi Metrik Evaluasi/Uji (`aggregate_evaluate`)
Metode ini dijalankan di server setelah client menyelesaikan pengujian lokal mereka:
* **Fungsi:** `MLflowStrategy.aggregate_evaluate(self, rnd, results, failures)`
* **Proses Agregasi:**
  1. Menerima skor loss dan metrik lain (seperti akurasi lokal) dari tiap client.
  2. Menggabungkan metrik tersebut dengan rumus rata-rata tertimbang berdasarkan jumlah sampel uji masing-masing client.
  3. Mencatat hasil agregasi akhir ke dashboard MLflow dengan nama `fedavg_eval_loss` dan `fedavg_eval_accuracy`.

---

## 4. Aspek Komparasi dalam Konteks Penelitian (Rekomendasi)

Untuk kebutuhan analisis/komparasi dalam penulisan tugas akhir atau tesis, berikut adalah beberapa skenario komparasi yang sangat kuat secara akademis:

### A. Centralized ML vs Federated Learning (Analisis Utility Drop)
Bandingkan performa model yang dilatih dengan metode tradisional (pusatkan semua data ke server lalu train) dengan model hasil Federated Learning:
* **Metrik Utama:** Akurasi, F1-Score, dan Recall (sangat penting untuk diagnosis medis agar meminimalisir False Negative).
* **Tujuan:** Mengukur seberapa besar performa model menurun (*utility drop*) sebagai harga yang harus dibayar demi mendapatkan privasi data (*privacy guarantee*).

### B. Performa Global vs Performa Lokal Client
Bandingkan metrik pengujian global (`centralized_accuracy`) dengan performa pengujian lokal di masing-masing client (`fedavg_eval_accuracy` atau metrik individual client):
* **Tujuan:** Menganalisis apakah model hasil agregasi global bersifat adil (*fairness*) dan bekerja dengan baik untuk seluruh client, atau hanya condong menguntungkan client dengan jumlah data paling banyak.

### C. Pengaruh Distribusi Data (IID vs Non-IID)
Lakukan eksperimen perbandingan dengan membagi dataset client menjadi dua skenario:
1. **Skenario IID:** Data dibagi merata dengan proporsi kelas target (sakit/sehat) dan karakteristik yang homogen di setiap client.
2. **Skenario Non-IID:** Data dibagi secara heterogen (misal: Client 1 mayoritas pasien pria, Client 2 mayoritas pasien wanita, atau ketidakseimbangan kelas target yang ekstrem).
* **Tujuan:** Menganalisis ketahanan algoritma FedAvg terhadap variasi/heterogenitas data klinis di dunia nyata.

### D. Parameter Pelatihan Terdistribusi
Bandingkan performa model dengan memvariasikan beberapa parameter kunci:
* **Jumlah Client (`min_fit_clients`)**: Pengaruh jumlah partisipan terhadap stabilitas konvergensi model.
* **Jumlah Iterasi Lokal (`max_iter`) vs Jumlah Ronde (`num_rounds`)**: Apakah lebih baik melakukan iterasi lokal yang banyak dengan ronde sedikit, atau iterasi lokal sedikit dengan ronde yang banyak (terkait efisiensi komunikasi jaringan).

---

## 5. Perbedaan Metrik Durasi Fit (`fedavg_fit_duration` vs `max_fit_duration`)

Di dashboard MLflow server, terdapat dua jenis pencatatan durasi komputasi pelatihan lokal client per ronde:

### A. `fedavg_fit_duration`
Metrik ini menunjukkan **rata-rata durasi** dari proses pelatihan model lokal di seluruh client pada ronde tersebut.
* **Rumus:**
  $$\text{fedavg\_fit\_duration} = \frac{\sum (\text{durasi\_fit}_{client})}{Jml\_Client\_Aktif}$$
* **Kegunaan:** Memberikan gambaran performa rata-rata/umum dari hardware client yang berpartisipasi.

### B. `max_fit_duration`
Metrik ini menunjukkan **durasi terlama (maksimal)** di antara seluruh client yang melakukan pelatihan model lokal pada ronde tersebut.
* **Rumus:**
  $$\text{max\_fit\_duration} = \max(\text{durasi\_fit}_{client\_1}, \text{durasi\_fit}_{client\_2}, \dots)$$
* **Kegunaan:** Mengidentifikasi adanya **"Stragglers"** (client yang lambat karena hardware kurang mumpuni atau beban kerja CPU lain). Dalam sistem sinkron (seperti FedAvg standar), server harus menunggu client terlama selesai sebelum melangkah ke ronde berikutnya. Oleh karena itu, metrik ini menjadi representasi nyata dari durasi total satu ronde pelatihan di sisi server (jika latency jaringan diabaikan).
