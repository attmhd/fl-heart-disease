# Blackbox Testing — Sistem Federated Learning untuk Deteksi Penyakit Jantung

Dokumen ini berisi skenario pengujian *blackbox testing* yang mencakup seluruh fungsionalitas sistem **Heart Disease Federated Learning**. Pengujian dilakukan menggunakan dua antarmuka utama yang berjalan secara bersamaan:

- **Streamlit Dashboard** (`http://localhost:8501`) — antarmuka utama kontrol pelatihan, monitoring metrik, health check container, dan log sistem.
- **MLflow UI** (`http://localhost:5000`) — antarmuka verifikasi pencatatan eksperimen, hierarki run, parameter, metrik, artefak, dan registri model.

Kedua antarmuka digunakan secara komplementer: Streamlit sebagai portal operasional, dan MLflow UI sebagai sumber kebenaran (*ground truth*) pencatatan eksperimen.

---

## Lingkungan Pengujian

| Komponen | Nilai |
| :--- | :--- |
| Streamlit Dashboard URL | `http://localhost:8501` |
| MLflow Tracking UI URL | `http://localhost:5000` |
| Flower Server Port (gRPC) | `localhost:8080` |
| Loki Log Aggregator URL | `http://localhost:3100` |
| Container yang diuji | `fl_server`, `mlflow_server`, `client_1`, `client_2`, `loki`, `alloy` |
| Dataset Client 1 | `heart_disease_hungary_cleaned.csv` |
| Dataset Client 2 | `heart_disease_switzerland_cleaned.csv` |
| Dataset Server (Evaluasi Terpusat) | `heart_disease_cleveland_cleaned.csv` |

---

## 1. Training Control Panel

**Tabel 1. Test Case FLHD-01**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Training Control Panel — Parameter & Kontrol Pelatihan |
| Deskripsi | Memverifikasi fungsionalitas pengisian form parameter, validasi kompatibilitas solver/penalty, tombol aksi (Launch/Stop/Clear), dan pengunggahan dataset melalui antarmuka Training Control Panel di Streamlit. |
| Pre-conditions | Dashboard Streamlit (`demo_app.py`) telah berjalan dan dapat diakses via browser. Docker daemon aktif di host. |
| Dependensi | Layanan Streamlit dan Docker Compose stack dapat dijalankan dari host. |
| Post-conditions | Parameter pelatihan terkonfigurasi, proses FL terpicu, log diperbarui, dan dataset ter-overwrite tanpa crash. |

**Tabel 2. Pengujian FLHD-01**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Menjalankan pelatihan FL dengan parameter default | Rounds=5, Solver=saga, Max Iter=1, C=1.0, Penalty=l2, Min Fit=2, Min Evaluate=2, Fraction Fit=1.0 | Status berubah menjadi "RUNNING", area "Active Execution Logs" mulai terisi log inisiasi server secara real-time | | |
| 2 | Menjalankan pelatihan dengan parameter kustom | Rounds=10, Solver=lbfgs, Max Iter=5, C=0.50, Penalty=l2 | Pelatihan berjalan dengan parameter baru; log awal server merefleksikan konfigurasi yang diinputkan | | |
| 3 | Menghentikan proses pelatihan FL yang sedang aktif | Klik "Stop / Terminate Stack" saat status RUNNING | Seluruh container FL dihentikan; status berubah menjadi "IDLE"; log mencatat aksi terminasi | | |
| 4 | Validasi kompatibilitas solver dan penalty — saga | Pilih Solver = saga | Opsi Penalty yang tersedia: l2, l1, none | | |
| 5 | Validasi kompatibilitas solver dan penalty — liblinear | Pilih Solver = liblinear | Opsi Penalty yang tersedia: l2, l1 | | |
| 6 | Validasi kompatibilitas solver dan penalty — lbfgs/newton-cg | Pilih Solver = lbfgs atau newton-cg | Opsi Penalty yang tersedia: l2, none | | |
| 7 | Mengunggah dataset untuk Server | Target = "Server (heart_disease_cleveland_cleaned.csv)", upload CSV valid, klik "Save & Overwrite Dataset" | Pesan sukses muncul; file CSV berhasil di-overwrite di `data/processed/heart_disease_cleveland_cleaned.csv` | | |
| 8 | Mengunggah dataset untuk Client 1 | Target = "Client 1 (heart_disease_hungary_cleaned.csv)", upload CSV valid | File `heart_disease_hungary_cleaned.csv` berhasil diperbarui; muncul pesan sukses | | |
| 9 | Mengunggah dataset untuk Client 2 | Target = "Client 2 (heart_disease_switzerland_cleaned.csv)", upload CSV valid | File `heart_disease_switzerland_cleaned.csv` berhasil diperbarui; muncul pesan sukses | | |
| 10 | Membersihkan log eksekusi aktif | Klik "Clear Logs" | Konten "Active Execution Logs" dikosongkan dan digantikan teks "=== Logs cleared by User ===" | | |
| 11 | Konfigurasi Fraction Fit | Ubah Fraction Fit = 0.5, klik Launch | Log server menunjukkan hanya 50% client yang disampling untuk ronde fit | | |
| 12 | Konfigurasi minimum client fit dan evaluate | Atur Min Fit Clients = 1, Min Evaluate Clients = 1, klik Launch | Pelatihan dapat berjalan meskipun hanya 1 client aktif | | |

---

## 2. Proses Koneksi dan Pelatihan Federasi

**Tabel 3. Test Case FLHD-02**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Koneksi gRPC Flower dan Proses Pelatihan Federasi |
| Deskripsi | Memverifikasi inisiasi koneksi gRPC antara FL client dan FL server, serta dimulainya ronde pelatihan pertama yang terpantau dari log di Streamlit. |
| Pre-conditions | Tombol "Launch Federated Training" telah diklik dan seluruh container sedang diinisiasi. |
| Dependensi | Docker network `fl_network` aktif; port 8080 dapat dijangkau oleh client_1 dan client_2. |
| Post-conditions | Koneksi gRPC antara client dan server terjalin; ronde pertama FL dimulai dan terlog. |

**Tabel 4. Pengujian FLHD-02**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Memantau log inisiasi FL Server | Panel "Active Execution Logs" di Streamlit | Log menampilkan pesan "Flower server running" atau "Starting gRPC server on 0.0.0.0:8080" | | |
| 2 | Memantau log koneksi FL Client 1 | "Active Execution Logs" atau tab "System Logs" difilter ke container `client_1` | Log menampilkan konfirmasi koneksi ke `fl_server:8080` via gRPC | | |
| 3 | Memantau log koneksi FL Client 2 | Tab "System Logs" difilter ke container `client_2` | Log menampilkan konfirmasi koneksi ke `fl_server:8080` via gRPC | | |
| 4 | Mengamati proses ronde pertama | Min Fit Clients=2; kedua client terhubung | Server mendeteksi 2 client, memulai instruksi fit, dan log menampilkan proses komputasi lokal client | | |

---

## 3. Experiment Tracking — Streamlit Dashboard

**Tabel 5. Test Case FLHD-03**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Experiment Tracking via Streamlit Dashboard |
| Deskripsi | Memverifikasi bahwa data eksperimen yang dicatat oleh MLflow (metrik, parameter, run hierarchy) tersajikan dengan benar di tab "Federated Training Monitor" pada Streamlit. |
| Pre-conditions | Pelatihan FL telah selesai minimal 1 eksperimen penuh. MLflow server aktif di port 5000. |
| Dependensi | Fungsi `fetch_metrics_from_mlflow()` dan `get_mlflow_evaluation_artifacts()` di `dashboard/data_provider.py`. |
| Post-conditions | Grafik Plotly, metric cards, dan artefak evaluasi tersaji lengkap dan akurat di tab monitor. |

**Tabel 6. Pengujian FLHD-03**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Memvalidasi metric card "Current FL Round" | Buka tab "Federated Training Monitor" pasca pelatihan | Card menampilkan nomor ronde terakhir yang selesai dieksekusi (misalnya: "5") | | |
| 2 | Memvalidasi metric card "Best Centralized Accuracy" | Amati card kedua di header dashboard | Card menampilkan persentase akurasi terpusat terbaik (contoh: "85.25%") | | |
| 3 | Memvalidasi metric card "FedAvg Val Accuracy" | Amati card ketiga di header dashboard | Card menampilkan akurasi validasi teragregasi FedAvg tertinggi yang dicapai | | |
| 4 | Memvalidasi grafik Training Accuracy | Buka tab "Federated Training Monitor" | Grafik garis Plotly menampilkan perkembangan akurasi training per-ronde untuk setiap client | | |
| 5 | Memvalidasi grafik Validation Accuracy | Buka tab "Federated Training Monitor" | Grafik menampilkan trace akurasi validasi per-client dan evaluasi sentral server per-ronde | | |
| 6 | Memvalidasi grafik Training Loss | Buka tab "Federated Training Monitor" | Grafik menampilkan tren loss training yang menurun seiring bertambahnya ronde | | |
| 7 | Memvalidasi grafik Validation Loss | Buka tab "Federated Training Monitor" | Grafik menampilkan tren loss validasi; tren naik mengindikasikan potensi overfitting | | |
| 8 | Memvalidasi grafik Training Duration | Bagian "Fit Duration per Round (ms)" | Grafik Plotly menampilkan durasi fit (`fedavg_fit_duration` dan `max_fit_duration`) per ronde dalam milidetik | | |
| 9 | Menguji filter multiselect Training Accuracy | Hapus salah satu pilihan di "Training Accuracy Metrics" | Trace garis metrik yang dihapus langsung menghilang secara interaktif dari grafik Plotly | | |
| 10 | Menguji filter multiselect Validation Loss | Hapus pilihan di "Validation Loss Metrics" | Trace garis yang tidak dipilih tersembunyi dari grafik | | |
| 11 | Verifikasi pesan saat belum ada data training | Buka dashboard sebelum ada pelatihan apapun | Muncul warning: "No federated training evaluation logs found yet. Start the Flower server and clients to begin streaming metrics." | | |
| 12 | Memvalidasi Confusion Matrix Server | Scroll ke bagian "MLflow Evaluation Artifacts" | Plot confusion matrix server termuat dengan heatmap, label "Sehat (0)" dan "Sakit Jantung (1)" | | |
| 13 | Memvalidasi Classification Report Server | Blok teks di bawah plot server | Laporan evaluasi menampilkan precision, recall, f1-score, dan support untuk kedua kelas | | |
| 14 | Memvalidasi Confusion Matrix & Report Client 1 & 2 | Kolom "Distributed CLIENT 1" dan "Distributed CLIENT 2" | Confusion matrix dan classification report masing-masing client tampil berdampingan (*side-by-side*) | | |

---

## 4. Experiment Tracking — MLflow UI

**Tabel 7. Test Case FLHD-04**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Experiment Tracking via MLflow UI (`http://localhost:5000`) |
| Deskripsi | Memverifikasi pencatatan eksperimen, hierarki Parent-Child Run, parameter model, metrik per ronde, dan artefak evaluasi langsung pada antarmuka web MLflow. |
| Pre-conditions | Pelatihan FL telah selesai. MLflow UI dapat diakses di `http://localhost:5000`. |
| Dependensi | Container `mlflow_server` aktif; volume `./mlruns` ter-mount dengan benar. |
| Post-conditions | Seluruh data eksperimen tersimpan, terstruktur, dan dapat ditelusuri di MLflow UI. |

**Tabel 8. Pengujian FLHD-04**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Mengakses MLflow UI | Buka `http://localhost:5000` di browser | Halaman MLflow UI termuat; daftar eksperimen tampil di panel kiri | | |
| 2 | Memvalidasi keberadaan eksperimen FL | Panel daftar eksperimen MLflow | Eksperimen bernama "FL_Heart_Disease" (atau sesuai konfigurasi) terdaftar dan dapat dibuka | | |
| 3 | Memvalidasi Parent Run server | Buka eksperimen, amati daftar run | Terdapat satu run utama (parent) dengan tag `mlflow.user = server` yang mewakili sesi federasi keseluruhan | | |
| 4 | Memvalidasi hierarki Child Run client | Klik parent run server, amati nested runs | Terdapat child run untuk `client_1` dan `client_2` yang terhubung ke parent run server via `mlflow.parentRunId` | | |
| 5 | Memvalidasi parameter FL Server | Klik parent run server, buka tab "Parameters" | Parameter `num_rounds` dan `strategy` tercatat dengan nilai yang sesuai konfigurasi pelatihan | | |
| 6 | Memvalidasi parameter model lokal Client | Klik salah satu child run client, buka tab "Parameters" | Parameter `solver`, `C`, `penalty`, dan `max_iter` tercatat sesuai nilai yang dikonfigurasi di dashboard | | |
| 7 | Memvalidasi metrik server per ronde | Parent run server → tab "Metrics" | Metrik `centralized_accuracy`, `centralized_loss`, `fedavg_train_accuracy`, `fedavg_val_accuracy`, `fedavg_fit_duration`, `max_fit_duration` terdaftar dan dapat diplot per ronde | | |
| 8 | Memvalidasi metrik client per ronde | Child run client → tab "Metrics" | Metrik `train_accuracy`, `train_loss`, `val_accuracy`, `val_loss`, `fit_duration` terdaftar per ronde untuk masing-masing client | | |
| 9 | Memvalidasi metrik best_model_round | Parent run server → tab "Metrics" | Metrik `best_model_round` tercatat sebagai nilai integer yang merepresentasikan ronde terbaik | | |
| 10 | Memvalidasi metrik best_model_val_accuracy | Parent run server → tab "Metrics" | Metrik `best_model_val_accuracy` tercatat dengan nilai akurasi validasi tertinggi yang dicapai selama pelatihan | | |
| 11 | Memvalidasi artefak Confusion Matrix di MLflow | Parent/child run → tab "Artifacts" | File gambar confusion matrix tersimpan sebagai artefak MLflow untuk server maupun setiap client | | |
| 12 | Memvalidasi artefak Classification Report di MLflow | Parent/child run → tab "Artifacts" | File teks classification report tersimpan sebagai artefak MLflow dan dapat dibaca melalui UI | | |
| 13 | Memvalidasi grafik metrik ronde di MLflow | Parent run → tab "Metrics" → klik salah satu metrik | Grafik tren metrik per ronde ditampilkan oleh MLflow UI secara interaktif | | |

---

## 5. Model Registry — MLflow UI

**Tabel 9. Test Case FLHD-05**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Model Registry via MLflow UI |
| Deskripsi | Memverifikasi registrasi model global terbaik, versi model, tag performa, dan alias promosi `@Production` pada MLflow Model Registry. |
| Pre-conditions | Pelatihan FL selesai dan proses registrasi model otomatis telah dieksekusi oleh server. |
| Dependensi | Container `mlflow_server` aktif; volume `./mlruns` ter-mount. |
| Post-conditions | Model global terdaftar dengan versi, tag, dan alias yang tepat di halaman "Models" MLflow UI. |

**Tabel 10. Pengujian FLHD-05**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Memvalidasi keberadaan model di registry | Buka `http://localhost:5000` → klik "Models" di navigasi atas | Model bernama "Heart_Disease_Global_Model" terdaftar di halaman Model Registry | | |
| 2 | Memvalidasi versi model terdaftar | Klik nama model di halaman Models | Terdapat minimal Version 1 yang terhubung ke run eksperimen FL | | |
| 3 | Memvalidasi tag performa model | Klik versi model → amati bagian "Tags" | Tag `validation_accuracy` dan `best_round` tercatat dengan nilai yang sesuai hasil pelatihan | | |
| 4 | Memvalidasi alias `@Production` | Halaman detail versi model | Alias `@Production` telah diberikan pada versi model dengan akurasi terbaik | | |
| 5 | Memvalidasi link ke parent run | Halaman detail versi model → klik "Source Run" | Link mengarah ke parent run server yang menghasilkan model tersebut | | |
| 6 | Memvalidasi format serialisasi skops | Halaman versi model → tab "Artifacts" | Artefak model menggunakan format serialisasi `skops` (bukan pickle standar), terindikasi dari ekstensi file atau metadata flavor | | |
| 7 | Memvalidasi incremental versi model | Jalankan pelatihan FL kedua kali dari Streamlit, lalu buka halaman Models | Model "Heart_Disease_Global_Model" kini memiliki Version 2; alias `@Production` berpindah ke versi terbaru dengan akurasi lebih baik | | |
| 8 | Memverifikasi konsistensi akurasi antara Streamlit dan MLflow | Bandingkan nilai "Best Centralized Accuracy" di Streamlit dengan tag `validation_accuracy` di MLflow Model Registry | Nilai akurasi yang ditampilkan di kedua antarmuka identik | | |

---

## 6. Model Registry — Representasi di Streamlit Dashboard

**Tabel 11. Test Case FLHD-06**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Model Registry — Representasi di Streamlit |
| Deskripsi | Memverifikasi bahwa informasi model terbaik dari MLflow Model Registry tersajikan secara akurat di metric card dan bagian artefak pada tab "Federated Training Monitor" Streamlit. |
| Pre-conditions | Pelatihan FL selesai; model global terdaftar di MLflow Registry. |
| Dependensi | Fungsi `fetch_metrics_from_mlflow()` di `dashboard/data_provider.py`. |
| Post-conditions | Metric card "Best Centralized Accuracy" dan artefak evaluasi di Streamlit konsisten dengan data di MLflow UI. |

**Tabel 12. Pengujian FLHD-06**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Memvalidasi penampilan akurasi terbaik | Amati metric card "Best Centralized Accuracy" di Streamlit | Nilai akurasi yang ditampilkan sesuai dengan `best_model_val_accuracy` yang tercatat di MLflow | | |
| 2 | Memvalidasi ronde terbaik | Amati card/info yang menampilkan ronde terbaik di Streamlit | Nilai ronde yang ditampilkan sesuai dengan `best_model_round` di MLflow | | |
| 3 | Memvalidasi confusion matrix terbaik | Bagian "MLflow Evaluation Artifacts" → kolom Server | Gambar confusion matrix yang tampil adalah artefak dari model global terbaik, bukan model ronde awal | | |
| 4 | Verifikasi pemuatan model `@Production` untuk evaluasi | Setelah pelatihan, amati log server di "Active Execution Logs" | Log server mencatat keberhasilan pemuatan model `@Production` untuk centralized evaluation final | | |

---

## 7. Infrastructure Health

**Tabel 13. Test Case FLHD-07**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Infrastructure Health — Monitoring Status Container |
| Deskripsi | Memverifikasi indikator status ONLINE/OFFLINE setiap container dan visualisasi topologi inter-service pada tab "Infrastructure Health" di Streamlit. |
| Pre-conditions | Dashboard Streamlit aktif dan tab "Infrastructure Health" dibuka. |
| Dependensi | Fungsi `get_docker_health_states()` di `dashboard/process_manager.py`; Docker daemon berjalan di host. |
| Post-conditions | Status dan topologi seluruh container divisualisasikan akurat sesuai kondisi aktual. |

**Tabel 14. Pengujian FLHD-07**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Memvalidasi status ONLINE semua container | Jalankan semua layanan via Streamlit, buka tab "Infrastructure Health" | Container `fl_server`, `mlflow_server`, `client_1`, `client_2`, `loki`, `alloy` menampilkan badge hijau "ONLINE" | | |
| 2 | Memvalidasi status OFFLINE semua container | Hentikan semua layanan via "Stop / Terminate Stack" | Seluruh container menampilkan badge merah "OFFLINE" | | |
| 3 | Memvalidasi status container individual | Jalankan hanya `mlflow_server` secara manual, buka tab "Infrastructure Health" | Hanya `mlflow_server` yang berstatus "ONLINE"; container lainnya "OFFLINE" | | |
| 4 | Memeriksa visualisasi topologi inter-service | Amati tabel "Inter-Service Topology" di tab Infrastructure Health | Tabel menampilkan relasi layanan (contoh: `client_1 → fl_server`, `fl_server → mlflow_server`) dengan status koneksi yang akurat | | |

---

## 8. System Logs (Loki Integration)

**Tabel 15. Test Case FLHD-08**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | System Logs — Loki Log Pipeline |
| Deskripsi | Memverifikasi pengumpulan log terpusat oleh Grafana Loki, fungsionalitas filter container dan tingkat severity, serta tampilan terminal virtual di Streamlit. |
| Pre-conditions | Dashboard Streamlit aktif; container `loki` dan `alloy` berjalan. |
| Dependensi | Grafana Loki (port 3100) dan Grafana Alloy sebagai log collector via Docker socket. |
| Post-conditions | Log dari seluruh container tersajikan secara real-time di terminal virtual Streamlit dengan filter yang berfungsi. |

**Tabel 16. Pengujian FLHD-08**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Memvalidasi log gabungan semua container | Dropdown "Filter by Container" = "ALL" | Terminal virtual menampilkan gabungan stream log dari semua container secara kronologis | | |
| 2 | Menguji filter log per container (client_1) | Pilih "client_1" pada dropdown filter container | Terminal hanya menampilkan log dari container `client_1` | | |
| 3 | Menguji filter log per container (fl_server) | Pilih "fl_server" pada dropdown filter container | Terminal hanya menampilkan log dari container `fl_server` | | |
| 4 | Menguji filter severity INFO | Pilih "Severity Level" = "INFO" | Terminal hanya menampilkan log informasi normal (tanpa baris warning/error) | | |
| 5 | Menguji filter severity WARNING | Pilih "Severity Level" = "WARNING" | Terminal hanya menampilkan baris log yang mengandung kata "warn" atau "warning" | | |
| 6 | Menguji filter severity ERROR/CRITICAL | Pilih "Severity Level" = "ERROR / CRITICAL" | Terminal hanya menampilkan baris log berisi "error", "fail", atau "critical" | | |
| 7 | Verifikasi tampilan jika log kosong | Pilih filter kombinasi yang tidak menghasilkan log apapun | Muncul pesan info: "No logs matched the selected filters. Awaiting telemetry stream from Loki." | | |
| 8 | Memeriksa limitasi baris log terminal | Jalankan FL yang memproduksi banyak log | Log di terminal virtual dibatasi (maksimum 150 atau 1000 baris terbaru) untuk mencegah kelambatan UI | | |

---

## 9. Telemetry Settings (Sidebar Streamlit)

**Tabel 17. Test Case FLHD-09**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Telemetry & Display Settings Sidebar |
| Deskripsi | Memverifikasi kustomisasi Loki Endpoint URL, fitur auto-refresh, dan interval refresh melalui sidebar Streamlit. |
| Pre-conditions | Dashboard Streamlit aktif. |
| Dependensi | Streamlit sidebar widgets (`st.text_input`, `st.checkbox`, `st.slider`). |
| Post-conditions | Konfigurasi sidebar dipropagasikan ke komponen dashboard dengan benar. |

**Tabel 18. Pengujian FLHD-09**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Mengubah Loki Endpoint URL ke URL valid | Ubah isian "Loki Endpoint URL" menjadi `http://localhost:3100` | Query log dashboard langsung menggunakan alamat Loki baru | | |
| 2 | Mengubah Loki Endpoint URL ke URL tidak valid | Ubah ke alamat tidak valid (misalnya: `http://invalid:9999`) | Log terminal tampil kosong tanpa crash; muncul pesan error yang informatif | | |
| 3 | Mengaktifkan fitur Auto-refresh | Centang "Auto-refresh logs" di sidebar | Dashboard melakukan pembaruan log dan metrik secara periodik otomatis | | |
| 4 | Menonaktifkan fitur Auto-refresh | Hapus centang "Auto-refresh logs" | Pembaruan otomatis berhenti; halaman hanya di-refresh saat ada interaksi user | | |
| 5 | Mengubah interval refresh | Geser slider "Refresh Interval" ke 5 detik | Siklus auto-refresh berjalan setiap 5 detik sekali | | |

---

## 10. Containerization & Docker Orchestration

**Tabel 19. Test Case FLHD-10**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Containerization & Docker Orchestration |
| Deskripsi | Memverifikasi urutan startup container sesuai dependensi healthcheck, propagasi variabel lingkungan (env vars), akses volume dataset, persistensi data MLflow, dan restart policy. |
| Pre-conditions | Dashboard diakses; stack Docker FL dijalankan dari control panel. |
| Dependensi | Docker Daemon pada host; konfigurasi `docker-compose.yml`. |
| Post-conditions | Urutan startup container benar, dataset ter-mount, dan data MLflow persisten pada host. |

**Tabel 20. Pengujian FLHD-10**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Verifikasi inisiasi container berhasil | Klik "Launch Federated Training" | Seluruh container berhasil di-start; status di dashboard berubah menjadi "RUNNING" tanpa hambatan build | | |
| 2 | Verifikasi urutan startup (dependensi healthcheck) | Amati tab "Infrastructure Health" di fase awal start | `mlflow_server` ONLINE terlebih dahulu, diikuti `fl_server`, lalu `client_1` dan `client_2` setelah fl_server healthy | | |
| 3 | Verifikasi propagasi env var FL_NUM_ROUNDS | Set "Number of FL Rounds" = 8, klik Launch | Log server mengonfirmasi pelatihan berjalan selama 8 ronde | | |
| 4 | Verifikasi propagasi env var MODEL_SOLVER | Set Solver = liblinear, klik Launch | Log client mengonfirmasi model menggunakan solver `liblinear` | | |
| 5 | Verifikasi akses volume dataset di container | Klik Launch FL Training | Client berhasil memuat dataset lokal dari volume; tidak ada error "File Not Found" | | |
| 6 | Verifikasi persistensi volume MLflow | Hentikan stack, jalankan kembali, buka MLflow UI | Riwayat eksperimen dan run lama tetap terbaca di `http://localhost:5000` | | |
| 7 | Verifikasi persistensi data di Streamlit | Hentikan stack, jalankan kembali, buka tab "Federated Training Monitor" | Data metrik eksperimen sebelumnya masih tersaji di grafik Streamlit | | |
| 8 | Verifikasi restart policy container | Observasi status container di tab "Infrastructure Health" jika terjadi gangguan sesaat | Container yang sempat terhenti mendadak otomatis pulih kembali berkat policy `restart: on-failure` | | |

---

## 11. Model Training — Federated Logistic Regression

**Tabel 21. Test Case FLHD-11**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Model Training — Federated Logistic Regression |
| Deskripsi | Memverifikasi inisialisasi model Logistic Regression, mekanisme warm start antar ronde, penanganan kombinasi solver/penalty, dan kebenaran pencatatan distribusi kelas dataset client. |
| Pre-conditions | Pelatihan FL dijalankan via control panel. |
| Dependensi | Scikit-learn dan Flower framework pada container; MLflow tracking aktif. |
| Post-conditions | Pelatihan berjalan dengan parameter yang benar dan tercermin di grafik monitor Streamlit serta metrik MLflow. |

**Tabel 22. Pengujian FLHD-11**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Verifikasi inisialisasi parameter model | Solver=saga, C=1.0, Penalty=l2, Max Iter=1 | Log mengonfirmasi parameter Logistic Regression diinisialisasi dengan benar di awal training | | |
| 2 | Verifikasi mekanisme warm start | Jalankan FL 5 ronde, amati grafik "Training Accuracy" di Streamlit | Akurasi meningkat secara bertahap seiring ronde, membuktikan `warm_start=True` aktif (bobot tidak direset per ronde) | | |
| 3 | Verifikasi penanganan Penalty = none | Penalty=none, Solver=saga | Training berjalan sukses tanpa `ValueError`; parameter model terpetakan aman sebagai `penalty=None` | | |
| 4 | Verifikasi pencatatan distribusi kelas | Amati log inisiasi client di "Active Execution Logs" | Log merekam jumlah sampel dan distribusi kelas (class 0 & class 1) dari dataset masing-masing client | | |
| 5 | Verifikasi ronde 0 tidak tercatat sebagai metrik | Buka MLflow UI → parent run server → tab Metrics | Tidak ada metrik evaluasi sentral yang tercatat untuk ronde 0 (hanya ronde 1 ke atas) | | |

---

## 12. Observability — Loki & Alloy

**Tabel 23. Test Case FLHD-12**

| Item | Deskripsi |
| :--- | :--- |
| Test Title | Observability — Loki dan Alloy |
| Deskripsi | Memverifikasi pipeline pengiriman log oleh Grafana Alloy ke Loki dan ketersediaan data log di Streamlit. |
| Pre-conditions | Stack Docker compose observabilitas aktif (loki, alloy). |
| Dependensi | Grafana Alloy membaca Docker socket host; Loki menerima log di port 3100. |
| Post-conditions | Aliran telemetri log berhasil terkumpul dari Docker socket dan tersedia di Streamlit. |

**Tabel 24. Pengujian FLHD-12**

| No | Test Step | Test Data | Expected Result | Actual Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Verifikasi Loki menerima log container | Buka tab "System Logs" di Streamlit dengan filter = "ALL" | Terminal virtual menampilkan log dari container yang aktif; berarti Loki berhasil menerima stream log | | |
| 2 | Verifikasi Alloy mengumpulkan log dari Docker socket | Jalankan FL dari Streamlit, amati tab System Logs | Log aktivitas `fl_server` dan `client_1/2` mengalir real-time di terminal virtual Streamlit | | |
| 3 | Verifikasi auto-refresh log berjalan | Aktifkan "Auto-refresh logs" di sidebar, jalankan FL | Baris-baris log baru bertambah otomatis di tab "System Logs" tanpa refresh manual | | |

---

> **Catatan:** Kolom "Actual Result" dan "Status" (✅ Berhasil / ❌ Gagal) diisi pada saat pelaksanaan pengujian aktual.
