# BAB 4 HASIL DAN PEMBAHASAN

Bab ini memaparkan hasil implementasi dan pembahasan sistem *Federated Learning* untuk klasifikasi penyakit jantung yang telah dirancang pada bab sebelumnya. Pembahasan disusun secara sistematis mengikuti alur implementasi, mulai dari lingkungan pengembangan, realisasi komponen-komponen sistem, hingga pengujian fungsionalitas keseluruhan sistem secara menyeluruh (*end-to-end*).

---

## 4.1 Lingkungan Implementasi

Seluruh komponen sistem diimplementasikan dan diuji pada lingkungan pengembangan dengan spesifikasi perangkat lunak sebagai berikut.

**Tabel 4.1 Spesifikasi Lingkungan Implementasi**

| Komponen | Versi / Keterangan |
| :--- | :--- |
| Sistem Operasi | Linux (Ubuntu) |
| Bahasa Pemrograman | Python 3.12 |
| Framework Federated Learning | Flower (`flwr`) v1.13+ |
| Platform MLOps | MLflow v2.x |
| Model Klasifikasi | Scikit-learn — Logistic Regression |
| Antarmuka Dashboard | Streamlit |
| Orkestrasi Kontainer | Docker & Docker Compose |
| Log Pipeline | Grafana Loki + Grafana Alloy |
| Manajemen Paket | `uv` (Python package manager) |
| Versi Kontrol | Git + DVC (Data Version Control) |

Lingkungan pengembangan memisahkan seluruh dependensi Python ke dalam *virtual environment* yang dikelola menggunakan `uv`, sementara untuk eksekusi distribusi digunakan Docker Compose yang mendefinisikan enam layanan utama: `mlflow_server`, `fl_server`, `client_1`, `client_2`, `loki`, dan `alloy`.

---

## 4.2 Implementasi Struktur Kode Sumber

### 4.2.1 Organisasi Modular Kode Sumber

Kode sumber sistem diorganisasikan secara modular di dalam direktori `src/` untuk mendukung pemisahan kepentingan (*separation of concerns*) yang bersih antara logika bisnis, infrastruktur federasi, dan utilitas pendukung. Struktur direktori kode sumber adalah sebagai berikut.

```text
src/
├── client/
│   └── client.py          ← Entry point Flower ClientApp
├── server/
│   └── server.py          ← Entry point Flower ServerApp
├── configs/
│   └── config.py          ← Konfigurasi terpusat (IP, jumlah ronde, path dataset, MLflow)
├── models/
│   └── model.py           ← Inisialisasi arsitektur Logistic Regression
├── data/
│   └── dataset.py         ← Helper loading & preprocessing dataset UCI Heart Disease
└── utils/
    ├── server_utils.py    ← Implementasi MLflowStrategy & Centralized Evaluation
    ├── client_utils.py    ← Implementasi HeartDiseaseClient & NumPyClient
    ├── data_utils.py      ← Statistik distribusi kelas data per-client
    ├── model_utils.py     ← Ekstraksi, injeksi & konversi parameter model
    └── eval_utils.py      ← Kalkulasi metrik evaluasi & pembuatan visualisasi artefak
```

Arsitektur ini menerapkan prinsip **High-Level Orchestration**, di mana berkas `server.py` dan `client.py` bertindak murni sebagai pengorkestrasi tingkat tinggi yang bersih, sementara seluruh logika bisnis—kalkulasi bobot, *callback* federasi, dan *tracking*—didelegasikan secara modular ke berkas *helper* di direktori `utils/`.

### 4.2.2 Konfigurasi Terpusat (`config.py`)

Seluruh parameter yang dapat dikonfigurasi oleh pengguna—termasuk alamat server FL, jumlah ronde, jalur dataset, URI MLflow, dan parameter model—dipusatkan pada modul `src/configs/config.py`. Pendekatan ini memungkinkan propagasi konfigurasi melalui variabel lingkungan (*environment variables*) Docker Compose ke dalam kontainer tanpa perubahan kode, sehingga mendukung eksperimentasi yang fleksibel melalui antarmuka Streamlit.

---

## 4.3 Implementasi Server Federated Learning

### 4.3.1 Inisialisasi Parent Run MLflow di Server (`server.py`)

Berkas `src/server/server.py` merupakan titik masuk (*entry point*) aplikasi server Flower. Fungsi `server_fn` dipanggil oleh *framework* Flower pada saat inisialisasi dan bertanggung jawab membuka *Parent Run* MLflow yang membungkus keseluruhan sesi pelatihan federasi. Alur inisialisasi adalah sebagai berikut.

1. **Pembuatan Parent Run:** Server membuka sebuah MLflow run baru dengan nama yang menyertakan *timestamp* eksekusi (contoh: `Federated_Server_Logistic_Regression_20250614_030000`). Identitas run (`server_run_id`) yang bersifat unik disimpan untuk kemudian didistribusikan ke seluruh client.
2. **Penetapan Metadata Run:** Server menetapkan sejumlah tag pada run MLflow untuk keperluan identifikasi, antara lain `mlflow.user = "server"`, `handled_by = "Admin Server"`, dan `project = "Federated Heart Disease"`.
3. **Pencatatan Parameter Awal:** Parameter global sesi pelatihan, yaitu `num_rounds` dan `strategy = "FedAvg"`, dicatat ke MLflow sebelum proses training dimulai.
4. **Pembangunan Strategi:** Fungsi `build_strategy_with_mlflow(server_run_id)` dari modul `server_utils.py` dipanggil untuk membangun objek strategi `MLflowStrategy` beserta konfigurasi *callback*-nya.

### 4.3.2 Strategi Agregasi Kustom (`MLflowStrategy`)

Inti dari sistem *experiment tracking* ini adalah kelas `MLflowStrategy` yang didefinisikan di `src/utils/server_utils.py`. Kelas ini memperluas (*extends*) kelas `fl.server.strategy.FedAvg` bawaan Flower dengan menambahkan orkestrasi pencatatan MLflow pada tiga titik kunci dalam siklus federasi.

#### A. Metode `aggregate_fit`

Metode ini dipanggil secara otomatis oleh *framework* Flower setiap kali server selesai menerima dan mengagregasi parameter model dari seluruh client yang berpartisipasi pada suatu ronde. Tahapan yang dilakukan adalah:

1. Memanggil `super().aggregate_fit()` untuk menjalankan algoritma agregasi bobot FedAvg asli dari Flower, yang mengimplementasikan rumus rata-rata tertimbang:

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot w_{t+1}^k$$

di mana $n_k$ adalah jumlah sampel data client $k$ dan $N = \sum n_k$ adalah total sampel seluruh client pada ronde tersebut.

2. Menyimpan salinan (*deep copy*) bobot agregasi saat ini sebagai `current_ndarrays`.
3. Menghitung dan mencatat ke MLflow sejumlah metrik agregasi per ronde, yaitu:
   - `fedavg_train_accuracy` — rata-rata tertimbang akurasi pelatihan lokal client
   - `fedavg_train_loss` — rata-rata tertimbang *loss* pelatihan lokal client
   - `fedavg_val_accuracy` — rata-rata tertimbang akurasi validasi lokal client
   - `fedavg_val_loss` — rata-rata tertimbang *loss* validasi lokal client
   - `fedavg_fit_duration` — rata-rata durasi komputasi lokal seluruh client (dalam milidetik)
   - `max_fit_duration` — durasi komputasi terlama di antara seluruh client (*straggler indicator*)

#### B. Metode `evaluate` (Evaluasi Terpusat)

Metode ini mengimplementasikan evaluasi terpusat (*centralized evaluation*) yang dieksekusi oleh server menggunakan dataset ujinya sendiri setelah setiap ronde agregasi. Ronde 0 dikecualikan secara eksplisit untuk menghindari pencatatan performa model yang belum terlatih. Metrik `centralized_accuracy` yang dihasilkan juga digunakan sebagai sinyal seleksi model terbaik: apabila nilai `centralized_accuracy` pada ronde saat ini melampaui nilai terbaik sebelumnya (`best_accuracy`), maka server menyimpan salinan bobot agregasi tersebut sebagai `best_ndarrays` beserta nomor ronde asalnya (`best_round`).

#### C. Metode `configure_evaluate` (Distribusi Model Terbaik)

Metode ini mengontrol parameter apa yang dikirimkan server kepada client pada fase evaluasi. Pada ronde terakhir (`server_round == FL_NUM_ROUNDS`), alih-alih mengirimkan bobot agregasi ronde terakhir, server mendistribusikan `best_ndarrays`—bobot model dari ronde dengan akurasi validasi terpusat tertinggi—kepada seluruh client untuk dievaluasi secara lokal. Mekanisme ini memastikan bahwa hasil evaluasi akhir di sisi client mencerminkan performa model global **terbaik**.

#### D. Metode `aggregate_evaluate`

Metode ini dipanggil setelah server mengumpulkan laporan evaluasi lokal dari seluruh client. Server menggabungkan metrik evaluasi lokal tersebut menggunakan rata-rata tertimbang dan mencatatnya sebagai `fedavg_eval_accuracy` dan `fedavg_eval_loss` di MLflow. Pada akhir ronde terakhir, metode ini juga memicu pemanggilan `_log_best_model_to_mlflow()` untuk menjalankan alur registrasi model otomatis.

### 4.3.3 Registrasi dan Promosi Model ke MLflow Registry

Fungsi `_log_best_model_to_mlflow()` di `server_utils.py` mengimplementasikan alur otomatisasi siklus deployment model global terbaik melalui tahapan berikut.

1. **Rekonstruksi Model:** Bobot NDArray terbaik (`best_ndarrays`) diinjeksikan kembali ke dalam instance model `LogisticRegression` baru menggunakan fungsi `model_utils.set_model_parameters()`.
2. **Registrasi ke MLflow:** Model didaftarkan ke MLflow Model Registry menggunakan `mlflow.sklearn.log_model()` dengan format serialisasi `skops`. Format `skops` dipilih karena lebih aman dibandingkan format `pickle` standar, karena menyertakan mekanisme validasi *schema* yang mencegah serangan *arbitrary code execution* melalui deserialisasi.
3. **Evaluasi Akhir Model Terbaik:** Apabila dataset server tersedia, model terbaik dievaluasi menggunakan fungsi `evaluate_and_log_metrics()` yang menghasilkan artefak *confusion matrix* dan *classification report* yang disimpan di MLflow.
4. **Penetapan Tag dan Alias:** Melalui `MlflowClient`, sistem secara otomatis menetapkan tag performa (`validation_accuracy`, `best_round`, `algorithm`, `environment`, `total_participating_clients`) serta mempromosikan versi model dengan alias `@champion` dan `@Production`.

---

## 4.4 Implementasi Client Federated Learning

### 4.4.1 Kelas `HeartDiseaseClient`

Kelas `HeartDiseaseClient` di `src/utils/client_utils.py` mengimplementasikan antarmuka `fl.client.NumPyClient` dari Flower. Kelas ini mengelola keseluruhan siklus pelatihan dan evaluasi lokal pada masing-masing node client.

### 4.4.2 Metode `fit` — Pelatihan Lokal dan Pencatatan Child Run

Alur eksekusi `fit()` pada setiap ronde adalah sebagai berikut.

1. **Penerimaan Bobot Global:** Server mengirimkan bobot global saat ini kepada client. Bobot tersebut diinjeksikan ke dalam model lokal menggunakan `model_utils.set_model_parameters()`.
2. **Pelatihan Lokal:** Model lokal dilatih menggunakan dataset lokal client selama `max_iter` iterasi dengan `warm_start=True`. Opsi `warm_start=True` memastikan model tidak direset ke kondisi awal pada setiap ronde, melainkan melanjutkan optimasi dari bobot terakhir yang diterima, sehingga memungkinkan konvergensi yang lebih cepat seiring bertambahnya ronde.
3. **Kalkulasi Metrik Lokal:** Setelah pelatihan, client menghitung metrik pada data latih dan data validasi lokal: `train_accuracy`, `train_loss`, `train_precision`, `train_recall`, `val_accuracy`, `val_loss`, `val_precision`, `val_recall`, dan `fit_duration` (durasi pelatihan dalam milidetik).
4. **Pembuatan Child Run MLflow:** Apabila belum ada *run* MLflow yang aktif, client membuat *Child Run* baru dengan menyertakan tag `{"mlflow.parentRunId": server_run_id}`. `server_run_id` ini diterima melalui kamus konfigurasi instruksi `fit()` yang dikirim oleh server. Mekanisme ini secara otomatis menautkan semua aktivitas pencatatan client ke dalam hierarki *Parent Run* server yang sama di dashboard MLflow.
5. **Pencatatan Parameter Sekali:** Pada ronde pertama (`current_round <= 1`), client mencatat parameter model (`solver`, `C`, `penalty`, `max_iter`) dan statistik dataset lokal (`num_samples`, `class_0`, `class_1`) ke MLflow.
6. **Pengiriman ke Server:** Client mengembalikan bobot model lokal yang telah diperbarui beserta jumlah sampel data latih dan seluruh metrik lokal kepada server.

### 4.4.3 Metode `evaluate` — Evaluasi Lokal dan Evaluasi Akhir

Pada fase evaluasi biasa, client hanya mengembalikan metrik `loss` dan `accuracy` dari data validasi lokal kepada server. Namun, pada ronde **terakhir** (`is_final_round = True`), client menjalankan prosedur evaluasi komprehensif:

1. Client mengaktifkan kembali *Child Run* MLflow-nya menggunakan `run_id` yang tersimpan (`mlflow_run_id`).
2. Client memanggil `eval_utils.evaluate_and_log_metrics()` yang menghasilkan dan mengunggah ke MLflow: *Classification report* dalam format teks (`.txt`) dan JSON (`.json`), serta plot *Confusion Matrix* dalam format gambar PNG.

Evaluasi ini dilakukan terhadap model global **terbaik** yang telah didistribusikan server, sehingga artefak evaluasi yang tersimpan di MLflow mencerminkan performa model yang akan dipromosikan ke `@Production`.

---

## 4.5 Implementasi Utilitas Evaluasi (`eval_utils.py`)

Fungsi `evaluate_and_log_metrics()` di `src/utils/eval_utils.py` merupakan fungsi utilitas terpusat untuk evaluasi model yang digunakan baik oleh server (evaluasi terpusat) maupun client (evaluasi lokal akhir). Fungsi ini melakukan tahapan berikut.

1. Menjalankan prediksi menggunakan `model.predict()` dan `model.predict_proba()`.
2. Menghitung `log_loss`, `accuracy_score`, `precision_score`, dan `recall_score`.
3. Mencatat seluruh metrik tersebut ke MLflow dengan *prefix* yang dapat dikonfigurasi (`metrics_prefix`), sehingga metrik server menggunakan prefix `centralized_` dan metrik client menggunakan prefix `best_model_client_`.
4. Apabila `artifact_path` disediakan, fungsi ini juga membuat dan menyimpan:
   - File teks *classification report* (`classification_report.txt`)
   - File JSON *classification report* (`classification_report.json`)
   - Plot *confusion matrix* menggunakan *heatmap* Seaborn dengan label kelas "Sehat (0)" dan "Sakit Jantung (1)" (`confusion_matrix.png`)
5. Seluruh artefak diunggah ke MLflow menggunakan `mlflow.log_artifact()` dengan memanfaatkan direktori sementara (`tempfile.TemporaryDirectory`) untuk meminimalisir penggunaan *disk* permanen di luar MLflow.

Penggunaan `matplotlib.use("Agg")` secara eksplisit memastikan pembuatan plot berjalan dalam mode *headless* (tanpa *display* grafis), yang merupakan prasyarat untuk eksekusi di lingkungan kontainer Docker.

---

## 4.6 Implementasi Orkestrasi Docker Compose

Seluruh layanan sistem diorkestrasi menggunakan Docker Compose yang mendefinisikan jaringan layanan `fl_network` dengan enam kontainer.

### 4.6.1 Urutan Startup dengan Healthcheck

Docker Compose mengimplementasikan mekanisme *healthcheck* untuk memastikan urutan startup yang benar dan mencegah *race condition*. Urutan startup yang dijamin oleh konfigurasi `depends_on` dengan kondisi `service_healthy` adalah sebagai berikut.

1. **`mlflow_server`** dijalankan pertama. Healthcheck menguji konektivitas ke port 5000 setiap 3 detik dengan 15 kali percobaan ulang.
2. **`fl_server`** baru dijalankan setelah `mlflow_server` dinyatakan *healthy*. Healthcheck server FL menguji konektivitas ke port 8080.
3. **`client_1`** dan **`client_2`** baru dijalankan setelah `fl_server` dinyatakan *healthy*.

Mekanisme ini memastikan bahwa server MLflow siap menerima koneksi sebelum server FL mencoba mencatat eksperimen, dan server FL siap menerima koneksi gRPC sebelum client mencoba terhubung.

### 4.6.2 Propagasi Konfigurasi via Environment Variables

Parameter pelatihan FL (seperti `FL_NUM_ROUNDS`, `MODEL_SOLVER`, `MODEL_C`, `MODEL_PENALTY`) didefinisikan sebagai variabel lingkungan (*environment variables*) yang diinjeksikan ke kontainer pada saat `docker compose up` dipanggil. Mekanisme ini memungkinkan Streamlit Dashboard mengubah konfigurasi pelatihan tanpa perlu membangun ulang *image* Docker.

### 4.6.3 Persistensi Data via Volume Mounting

Volume `./mlruns` di-*mount* ke semua kontainer yang berinteraksi dengan MLflow (`fl_server`, `client_1`, `client_2`, dan `mlflow_server`). Hal ini memastikan riwayat eksperimen dan artefak model persisten pada *host filesystem* dan tidak hilang ketika kontainer dihentikan atau dihapus.

### 4.6.4 Pipeline Observabilitas Log (Loki + Alloy)

Kontainer `alloy` (Grafana Alloy) di-*mount* langsung ke soket Docker host (`/var/run/docker.sock`) dalam mode *read-only*. Konfigurasi ini memungkinkan Alloy membaca *log stream* dari seluruh kontainer yang berjalan secara otomatis tanpa modifikasi kode Python apapun. Log yang dikumpulkan kemudian dikirim ke kontainer `loki` melalui port 3100, di mana Alloy secara otomatis menambahkan label `container` dari metadata Docker sebagai label yang dapat digunakan untuk memfilter log di Streamlit.

---

## 4.7 Implementasi Dashboard Streamlit

Dashboard utama sistem diimplementasikan menggunakan Streamlit dan dapat diakses di `http://localhost:8501`. Dashboard ini terdiri dari empat tab utama.

### 4.7.1 Tab Training Control Panel

Tab ini menyediakan antarmuka pengguna untuk mengkonfigurasi dan mengendalikan proses pelatihan FL. Komponen utama tab ini antara lain:

- **Form Parameter Pelatihan:** Input untuk `FL_NUM_ROUNDS`, `MODEL_SOLVER`, `MODEL_C`, `MODEL_PENALTY`, `MODEL_MAX_ITER`, `FL_FRACTION_FIT`, `FL_MIN_FIT_CLIENTS`, dan `FL_MIN_EVALUATE_CLIENTS`. Pemilihan kombinasi *solver* dan *penalty* divalidasi secara dinamis menggunakan logika kompatibilitas Scikit-learn.
- **Tombol Aksi:** `Launch Federated Training` untuk menjalankan `docker compose up`, `Stop / Terminate Stack` untuk `docker compose down`, dan `Clear Logs` untuk membersihkan tampilan log.
- **Area Log Eksekusi Aktif:** Terminal virtual yang menampilkan *output* log dari `docker compose` secara *real-time*.
- **Pengunggahan Dataset:** Antarmuka untuk mengunggah file CSV dataset baru untuk ketiga partisi (Server, Client 1, Client 2).

### 4.7.2 Tab Federated Training Monitor

Tab ini menampilkan visualisasi perkembangan pelatihan FL secara interaktif. Data metrik diambil terutama dari **MLflow** sebagai sumber data utama (*ground truth*) melalui fungsi `fetch_metrics_from_mlflow()` di `dashboard/data_provider.py`.

Fungsi ini menggunakan `MlflowClient` untuk mengambil riwayat metrik per-ronde dari MLflow API dan membangun sebuah DataFrame *pivot* yang menggabungkan metrik dari *parent run* server dan seluruh *child run* client dalam satu struktur data tabel. Tab ini menampilkan:

- **Metric Cards:** Menampilkan `Current FL Round`, `Best Centralized Accuracy`, dan `FedAvg Val Accuracy`.
- **Grafik Plotly Interaktif:** Grafik garis untuk Training Accuracy, Validation Accuracy, Training Loss, Validation Loss, dan Fit Duration per ronde, masing-masing dengan *multiselect filter*.
- **MLflow Evaluation Artifacts:** Menampilkan *confusion matrix* dan *classification report* dari model terbaik untuk server serta masing-masing client secara berdampingan.

### 4.7.3 Tab Infrastructure Health

Tab ini menampilkan status *online/offline* dari setiap kontainer dalam sistem menggunakan fungsi `get_docker_health_states()` di `dashboard/process_manager.py`. Tab ini juga menampilkan tabel topologi antar-layanan yang merepresentasikan alur komunikasi antar komponen sistem.

### 4.7.4 Tab System Logs

Tab ini menyajikan antarmuka terminal virtual yang menampilkan log dari seluruh kontainer secara terpadu. Log diambil dari Grafana Loki melalui API HTTP (`/loki/api/v1/query_range`) dengan kueri LogQL. Tersedia filter **by Container** dan **by Severity Level** (Info, Warning, Error/Critical), serta fitur *auto-refresh* periodik menggunakan mekanisme `st.fragment` dari Streamlit.

---

## 4.8 Analisis Experiment Tracking

### 4.8.1 Hierarki Parent-Child Run

Mekanisme *Parent-Child Run* yang diimplementasikan berhasil menciptakan struktur hierarki eksperimen yang koheren di MLflow. Pada setiap sesi pelatihan federasi, MLflow mencatat satu *parent run* server yang membungkus dua *child run* client. Hubungan hierarki ini diwujudkan melalui tag `mlflow.parentRunId` yang secara otomatis ditambahkan client ke MLflow pada saat membuat *child run*.

Tanpa mekanisme ini, metrik dari setiap client dan server akan tercatat sebagai *run* yang sepenuhnya terisolasi, sehingga perbandingan dan analisis lintas komponen menjadi sangat sulit dilakukan. Dengan hierarki yang terstruktur, peneliti dapat dengan mudah menelusuri aktivitas pelatihan dari level sesi secara keseluruhan (*parent run*) hingga detail level client individual (*child run*) dalam satu antarmuka MLflow yang terintegrasi.

### 4.8.2 Metrik yang Dicatat

Tabel berikut merangkum seluruh metrik yang berhasil dicatat oleh sistem pada setiap sesi pelatihan.

**Tabel 4.2 Daftar Metrik yang Dicatat di MLflow**

| Level Run | Nama Metrik MLflow | Deskripsi |
| :--- | :--- | :--- |
| **Parent (Server)** | `centralized_accuracy` | Akurasi evaluasi model global pada dataset uji server |
| **Parent (Server)** | `centralized_loss` | Log loss evaluasi model global pada dataset uji server |
| **Parent (Server)** | `centralized_precision` | Precision evaluasi model global (terpusat) |
| **Parent (Server)** | `centralized_recall` | Recall evaluasi model global (terpusat) |
| **Parent (Server)** | `fedavg_train_accuracy` | Rata-rata tertimbang akurasi training lokal client |
| **Parent (Server)** | `fedavg_train_loss` | Rata-rata tertimbang *loss* training lokal client |
| **Parent (Server)** | `fedavg_val_accuracy` | Rata-rata tertimbang akurasi validasi lokal client |
| **Parent (Server)** | `fedavg_val_loss` | Rata-rata tertimbang *loss* validasi lokal client |
| **Parent (Server)** | `fedavg_fit_duration` | Rata-rata durasi pelatihan lokal per ronde (ms) |
| **Parent (Server)** | `max_fit_duration` | Durasi pelatihan lokal terlama per ronde (ms) |
| **Parent (Server)** | `best_model_round` | Nomor ronde dengan akurasi terpusat terbaik |
| **Parent (Server)** | `best_model_val_accuracy` | Nilai akurasi terpusat terbaik yang dicapai |
| **Child (Client)** | `train_accuracy` | Akurasi pelatihan lokal per ronde |
| **Child (Client)** | `train_loss` | Log loss pelatihan lokal per ronde |
| **Child (Client)** | `train_precision` | Precision pelatihan lokal per ronde |
| **Child (Client)** | `train_recall` | Recall pelatihan lokal per ronde |
| **Child (Client)** | `val_accuracy` | Akurasi validasi lokal per ronde |
| **Child (Client)** | `val_loss` | Log loss validasi lokal per ronde |
| **Child (Client)** | `val_precision` | Precision validasi lokal per ronde |
| **Child (Client)** | `val_recall` | Recall validasi lokal per ronde |
| **Child (Client)** | `fit_duration` | Durasi pelatihan lokal per ronde (ms) per client |

### 4.8.3 Analisis Metrik Durasi Fit: Deteksi Straggler

Pencatatan dua metrik durasi yang berbeda—`fedavg_fit_duration` (rata-rata) dan `max_fit_duration` (maksimum)—merupakan mekanisme yang dirancang untuk mendeteksi *straggler* dalam sistem. *Straggler* adalah istilah dalam *Federated Learning* untuk menggambarkan client yang memiliki waktu komputasi lokal jauh lebih lama dibandingkan client lainnya.

Pada algoritma FedAvg yang bersifat sinkron, server harus menunggu seluruh client yang ditunjuk menyelesaikan pelatihan lokal sebelum dapat melanjutkan ke ronde berikutnya. Oleh karena itu, `max_fit_duration` secara efektif merepresentasikan waktu nyata satu ronde federasi dari perspektif server (mengabaikan latensi jaringan). Selisih yang besar antara `max_fit_duration` dan `fedavg_fit_duration` mengindikasikan adanya ketidakseimbangan beban komputasi antar client—kondisi yang akan semakin kritis jika jumlah client diperbesar dalam skenario produksi.

---

## 4.9 Analisis Model Registry

### 4.9.1 Seleksi Model Terbaik Otomatis

Sistem mengimplementasikan mekanisme seleksi model terbaik yang berjalan secara otomatis selama proses pelatihan berlangsung. Pada setiap pemanggilan `evaluate()`, server membandingkan `centralized_accuracy` ronde saat ini dengan nilai `best_accuracy` yang tersimpan. Apabila terjadi peningkatan akurasi, server menyimpan salinan (*deep copy*) bobot NDArray agregasi beserta nomor ronde yang bersangkutan. Pendekatan ini memastikan bahwa model yang dipromosikan ke *registry* adalah model dengan performa terpusat terbaik yang pernah dicapai selama seluruh sesi federasi, bukan sekadar model dari ronde terakhir.

### 4.9.2 Otomatisasi Siklus Deployment

Setelah proses pelatihan selesai, sistem secara otomatis menjalankan alur deployment model yang mencakup rekonstruksi, registrasi, evaluasi akhir, pemberian tag, dan promosi alias `@Production`. Alur ini meminimalkan proses manual yang rentan terhadap kesalahan manusia dan memungkinkan setiap sesi pelatihan menghasilkan versi model yang baru dan terlacak dalam MLflow Registry. Setiap eksekusi ulang pelatihan akan menghasilkan versi model yang bertambah secara inkremental (Version 1, Version 2, dst.) dengan alias `@Production` yang selalu mengacu pada versi dengan akurasi terbaik.

### 4.9.3 Keamanan Format Serialisasi `skops`

Model global didaftarkan ke MLflow menggunakan format serialisasi `skops` yang lebih aman dibandingkan `pickle` standar. Format `skops` memvalidasi struktur objek Python yang akan di-deserialisasi menggunakan sebuah *schema* yang telah disetujui sebelumnya, sehingga mencegah serangan *pickle injection* di mana penyerang menyisipkan kode berbahaya dalam file model yang telah dimanipulasi. Penggunaan `skops` merupakan praktik terbaik yang direkomendasikan untuk model ML yang digunakan dalam konteks produksi, terutama pada domain medis.

---

## 4.10 Pembahasan Strategi Evaluasi

### 4.10.1 Dua Strategi Evaluasi yang Diimplementasikan

Sistem mengimplementasikan dua strategi evaluasi model global yang memiliki *trade-off* berbeda antara objektivitas pengukuran dan kepatuhan prinsip privasi Federated Learning.

**Strategi 1: Evaluasi Terpusat (*Centralized Evaluation*)**

Pada strategi ini, model global dievaluasi oleh server menggunakan dataset uji yang disimpan di sisi server (`heart_disease_cleveland_cleaned.csv`). Keunggulan strategi ini adalah memberikan metrik performa yang **konsisten dan tidak bias** karena semua ronde dievaluasi menggunakan set data yang sama. Nilai `centralized_accuracy` dapat dibandingkan langsung antar ronde tanpa ambiguitas, sehingga dapat digunakan sebagai sinyal seleksi model terbaik yang andal.

Keterbatasan strategi ini adalah dalam skenario produksi klinis nyata, mengumpulkan data pasien ke server pusat untuk dijadikan dataset uji bertentangan dengan prinsip dasar Federated Learning. Dalam penelitian ini, dataset Cleveland dari UCI Heart Disease digunakan sebagai *proxy* dataset uji server dalam konteks prototipe penelitian. Untuk implementasi klinis nyata, alternatif yang dapat dipertimbangkan adalah penggunaan **data sintetis berkualitas tinggi** yang dibangkitkan oleh model generatif—seperti *Conditional Tabular GAN* (CTGAN)—sehingga server memiliki dataset uji yang merepresentasikan distribusi statistik populasi pasien tanpa memaparkan data rekam medis asli.

**Strategi 2: Evaluasi Terdistribusi (*Client-Side Evaluation*)**

Pada strategi ini, model global dievaluasi secara lokal di masing-masing client menggunakan data uji lokal mereka sendiri. Keunggulan strategi ini adalah sepenuhnya mematuhi prinsip privasi FL karena tidak ada data yang meninggalkan node lokal. Namun, metrik yang dihasilkan berpotensi mengandung **bias evaluasi lokal** akibat distribusi data *non-IID* antar client.

Pada skenario penelitian ini, distribusi kelas di dataset Hungaria (`client_1`) dan Swiss (`client_2`) memiliki karakteristik yang berbeda, sehingga metrik validasi lokal masing-masing client mencerminkan performa model pada sub-populasi tersebut, bukan pada populasi umum. Agregasi rata-rata tertimbang dari metrik yang berpotensi bias ini menyulitkan penarikan kesimpulan tunggal yang valid mengenai performa model global secara menyeluruh.

### 4.10.2 Pendekatan yang Dipilih

Dalam penelitian ini, **evaluasi terpusat dipilih sebagai metode evaluasi utama** dan sinyal seleksi model terbaik, sementara evaluasi terdistribusi berfungsi sebagai metrik pendamping untuk analisis keadilan (*fairness*) antar client. Keputusan ini didasarkan pada kebutuhan penelitian untuk memiliki sinyal performa yang objektif dan konsisten guna mendukung mekanisme seleksi model otomatis yang andal.

---

## 4.11 Hasil Pengujian Fungsionalitas Sistem (*Blackbox Testing*)

Pengujian fungsionalitas sistem dilakukan menggunakan metode *blackbox testing* yang mencakup 12 skenario pengujian (FLHD-01 sampai FLHD-12). Pengujian dilakukan secara terintegrasi menggunakan dua antarmuka utama: **Streamlit Dashboard** (`http://localhost:8501`) sebagai portal operasional dan **MLflow UI** (`http://localhost:5000`) sebagai sumber kebenaran (*ground truth*) pencatatan eksperimen.

Ringkasan skenario pengujian dan aspek yang diverifikasi disajikan pada tabel berikut.

**Tabel 4.3 Ringkasan Skenario Pengujian Blackbox**

| Kode Uji | Aspek yang Diuji | Antarmuka Pengujian |
| :--- | :--- | :--- |
| FLHD-01 | Parameter & Kontrol Pelatihan, Validasi Kompatibilitas Solver/Penalty, Unggah Dataset | Streamlit |
| FLHD-02 | Koneksi gRPC Flower & Inisiasi Ronde Pelatihan Pertama | Streamlit (Logs) |
| FLHD-03 | Experiment Tracking via Streamlit (Grafik Interaktif & Artefak Evaluasi) | Streamlit |
| FLHD-04 | Experiment Tracking via MLflow UI (Hierarki Run, Parameter, Metrik per Ronde) | MLflow UI |
| FLHD-05 | Model Registry via MLflow UI (Versi, Tag Performa, Alias `@Production`) | MLflow UI |
| FLHD-06 | Representasi Model Registry di Streamlit (Konsistensi Antar Antarmuka) | Streamlit & MLflow UI |
| FLHD-07 | Infrastructure Health — Status ONLINE/OFFLINE Container | Streamlit |
| FLHD-08 | System Logs — Filter Container & Severity via Loki | Streamlit |
| FLHD-09 | Telemetry Settings Sidebar (Endpoint Loki, Auto-refresh, Interval) | Streamlit |
| FLHD-10 | Containerization & Docker Orchestration (Urutan Startup, Env Var, Persistensi) | Streamlit & Terminal |
| FLHD-11 | Model Training — Federated Logistic Regression (Warm Start, Parameter Validity) | Streamlit & MLflow UI |
| FLHD-12 | Observabilitas — Loki & Alloy Log Pipeline | Streamlit |

> **Catatan:** Kolom "Actual Result" dan "Status" (Berhasil / Gagal) pada setiap tabel pengujian diisi pada saat pelaksanaan pengujian aktual. Rincian lengkap setiap skenario pengujian dapat dilihat pada lampiran dokumen *blackbox testing*.

---

## 4.12 Pembahasan Integrasi Sistem Secara Menyeluruh

Berdasarkan implementasi yang telah dipaparkan, sistem yang dibangun berhasil mengintegrasikan empat lapisan teknologi menjadi satu ekosistem yang koheren:

1. **Lapisan Federasi (Flower):** Menyediakan mekanisme komunikasi gRPC yang aman antara server dan client, serta mengimplementasikan algoritma FedAvg dengan *callback strategy* yang dapat dikustomisasi.

2. **Lapisan Pelacakan Eksperimen (MLflow):** Menyediakan penyimpanan terpusat untuk seluruh metadata eksperimen (parameter, metrik, artefak) dengan hierarki *Parent-Child Run* yang mencerminkan struktur alami sistem FL.

3. **Lapisan Operasional (Streamlit):** Menyediakan antarmuka pengguna tunggal yang mengintegrasikan kontrol pelatihan, pemantauan metrik *real-time*, pemantauan kesehatan infrastruktur, dan tampilan log sistem.

4. **Lapisan Observabilitas (Loki + Alloy):** Menyediakan pipeline log yang mengumpulkan, mengagregasi, dan mengekspos log dari seluruh kontainer Docker tanpa modifikasi kode sumber, sehingga log tersebut dapat ditampilkan di Streamlit dalam satu antarmuka terpadu.

Kombinasi keempat lapisan ini menjawab kesenjangan penelitian yang diidentifikasi: aspek *MLOps* dan operasional *Federated Learning* yang sering diabaikan dalam literatur FL yang berfokus pada algoritma. Sistem yang dibangun memberikan visibilitas penuh terhadap seluruh proses pelatihan terdistribusi—mulai dari parameter konfigurasi, metrik per-ronde per-client, hingga status infrastruktur dan aliran log—dalam satu lingkungan yang dapat direproduksi menggunakan Docker Compose, tanpa melanggar prinsip privasi data *Federated Learning*.

---

> *Catatan Penyelesaian: Draft bab ini perlu dilengkapi dengan tangkapan layar (screenshot) dari antarmuka Streamlit Dashboard dan MLflow UI pada setiap tahap pengujian, serta tabel hasil aktual pengujian blackbox setelah pengujian dilaksanakan.*
