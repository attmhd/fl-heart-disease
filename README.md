# Experiment Tracking and Model Registry on Heart Disease Classification using MLflow and Flower

Proyek ini merupakan sistem pembelajaran terfederasi (**Federated Learning**) tingkat lanjut untuk klasifikasi penyakit jantung menggunakan dataset **UCI Heart Disease**. Sistem ini mengintegrasikan framework **Flower** untuk koordinasi server-client, **MLflow** untuk *experiment tracking* & *model registry* tingkat lanjut, serta **Grafana Loki Stack** untuk observabilitas dan pengumpulan log kontainer secara tersentralisasi.

---

## 📐 Arsitektur Sistem & Alur Kerja

Berikut adalah visualisasi arsitektur sistem pembelajaran terfederasi beserta sistem *tracking* dan *monitoring* terintegrasi:

```mermaid
graph TB
    %% Nodes & Structure
    subgraph ClientLayer ["🍀 Client Zone (Local Training)"]
        C1["💻 Client 1 (client_1 container)"]
        C2["💻 Client 2 (client_2 container)"]
        D1[("💾 Partisi Data 1 (Hungaria)")]
        D2[("💾 Partisi Data 2 (Swiss)")]
        C1 --- D1
        C2 --- D2
    end

    subgraph FLServerLayer ["🧠 Federated Server Zone"]
        FS["⚙️ Flower Server (fl_server container)"]
        FA["⚖️ FedAvg Aggregator"]
        FS --- FA
    end

    subgraph TrackingLayer ["📈 MLflow Tracking & Model Registry"]
        MLF["📊 MLflow Tracking Server"]
        MR[("🗂️ Model Registry (Stage: Production)")]
        MLF --- MR
    end

    subgraph MonitoringLayer ["📊 Centralized Observability Stack"]
        PT["🕷️ Promtail Log Shipper"]
        LK[("🪵 Grafana Loki (Log DB)")]
        GF["📈 Grafana UI Dashboard (Port 3000)"]
        PT --> LK
        LK --> GF
    end

    %% Connections & Protocols
    C1 <== "gRPC (Port 8080)" ==> FS
    C2 <== "gRPC (Port 8080)" ==> FS

    FS ==>|"Log Parameter & Agregasi Metrik"| MLF
    C1 ==>|"Log Metrik Lokal & Bobot Model"| MLF
    C2 ==>|"Log Metrik Lokal & Bobot Model"| MLF

    %% Log Scraping via Docker Socket
    DS["🐳 Docker Engine Daemon (/var/run/docker.sock)"]
    C1 -. "Streams Stdout/Stderr" .-> DS
    C2 -. "Streams Stdout/Stderr" .-> DS
    FS -. "Streams Stdout/Stderr" .-> DS
    MLF -. "Streams Stdout/Stderr" .-> DS

    DS ==>|"Read & Relabel Logs"| PT
    PT ==>|"Push Structured Logs (Port 3100)"| LK

    %% Styling
    classDef client fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef server fill:#efebe9,stroke:#5d4037,stroke-width:2px;
    classDef tracking fill:#ede7f6,stroke:#5e35b1,stroke-width:2px;
    classDef monitor fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;

    class C1,C2 client;
    class FS,FA server;
    class MLF tracking;
    class PT,GF monitor;
    class D1,D2,MR,LK,DS storage;
```

---

## ⚡ Fitur Utama

### 1. Federated Learning Tingkat Lanjut (Flower + FedAvg)
* Orkestrasi pembelajaran terfederasi menggunakan **Flower Framework** secara aman tanpa membagikan data mentah client.
* Menggunakan algoritma **Federated Averaging (FedAvg)** untuk menggabungkan bobot model lokal menjadi model global.

### 2. Pengukuran Bobot Model & Pembuktian FedAvg
* **Tanpa Random Hash**: Client ID bawaan Flower yang acak diabaikan dan diganti dengan indeks sequential bersih (`client_1`, `client_2`, dst.) agar visualisasi di MLflow konsisten dan rapi.
* **Metrik Bobot Model**: Mengukur nilai L2 norm dari bobot lokal model (`param_norm`) dan perbedaan pembaruan bobot lokal terhadap model global sebelumnya (`delta_norm`) sebelum dikirim ke server. Ini memberikan transparansi penuh untuk pembuktian rumus matematika FedAvg.

### 3. MLflow Tracking & Model Registry Otomatis
* **Standardized Run Naming**: Penamaan run terstruktur secara otomatis: `Federated_Server_Logistic_Regression_[Timestamp]` dan `Federated_Client_[ID]_Logistic_Regression_[Timestamp]`.
* **Best Model Registry**: Hanya model global terbaik (berdasarkan akurasi validasi teragregasi tertinggi lintas seluruh ronde) yang didaftarkan ke *Model Registry*, menghindari sampah versi model per ronde.
* **Production Promotion**: Model terbaik secara otomatis ditransisikan ke tahap `"Production"`, diberikan metadata tags lengkap (`validation_accuracy`, `best_round`, `algorithm`, `total_participating_clients`), serta ditautkan ke alias `"champion"` dan `"Production"` menggunakan `MlflowClient`. Versi model lama diarsipkan secara otomatis.

### 4. Observabilitas Tersentralisasi (Grafana Loki & Promtail Stack)
* Pengumpulan log aktif dan otomatis dari seluruh container (`fl_server`, `client_1`, `client_2`, `mlflow`) dengan menghubungkan soket Docker `/var/run/docker.sock`.
* Dashboard **Docker Container Logs** siap pakai di Grafana (Port 3000) untuk mempermudah pencarian log, visualisasi volume log, filter container, dan monitoring *error* / *warning* secara real-time.

---

## 🔌 Pemetaan Port Layanan

| Layanan | Port Eksternal (Host) | Port Internal (Container) | Protokol | Deskripsi |
| :--- | :--- | :--- | :--- | :--- |
| **Grafana** | `3000` | `3000` | HTTP | Dashboard Visualisasi Log & Monitoring |
| **Loki** | `3100` | `3100` | HTTP | Basis data log tersentralisasi |
| **MLflow Server** | `5000` | `5000` | HTTP | Experiment Tracking & Model Registry |
| **FL Server (Flower)** | `8080` | `8080` | TCP (gRPC) | Server Pembelajaran Terfederasi |
| **Promtail** | *N/A* | `9080` | HTTP | Pengumpul log container Docker |

---

## 📂 Struktur Proyek

```text
.
├── configs/            # Konfigurasi global (MLflow, Flower, dsb)
├── data/               # Dataset mentah dan hasil pemrosesan
│   ├── raw/
│   └── processed/
├── src/                # Source code utama
│   ├── client/         # Implementasi Flower Client & kalkulasi bobot L2
│   ├── server/         # Implementasi Flower Server, FedAvg & Registry Automation
│   ├── models/         # Arsitektur model (Scikit-learn Logistic Regression)
│   └── utils/          # Fungsi pembantu (data loading, preprocessing)
├── monitoring/         # Konfigurasi Grafana Loki & Promtail
│   ├── grafana/        # Dashboards & Datasources provisioning
│   └── loki/           # Konfigurasi database Loki
├── Dockerfile.client   # Dockerfile teroptimasi (slim size dengan Astral UV)
├── Dockerfile.server   # Dockerfile teroptimasi (slim size dengan Astral UV)
├── docker-compose.yml  # Orkestrasi seluruh layanan FL + Observabilitas
└── pyproject.toml      # Manajemen dependensi Python & konfigurasi Ruff/Black
```

---

## 🛠️ Prasyarat

Sebelum memulai, pastikan sistem Anda telah menginstal:
* [Python 3.11+](https://www.python.org/)
* [UV](https://github.com/astral-sh/uv) (Pengelola paket Python cepat)
* [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)

---

## ⚙️ Cara Menjalankan

### 1. Menggunakan Docker Compose (Sangat Direkomendasikan)

Ini adalah cara tercepat untuk menjalankan seluruh ekosistem FL beserta sistem monitoring log tersentralisasi:

```bash
docker-compose up --build
```

Setelah semua kontainer aktif:
- **MLflow Dashboard**: Buka [http://localhost:5000](http://localhost:5000) untuk melihat eksperimen.
- **Grafana Loki Logs**: Buka [http://localhost:3000](http://localhost:3000) (User: `admin` | Sandi: `admin`).
  - Pilih menu **Dashboards** -> folder **Monitoring** -> pilih dashboard **Docker Container Logs** untuk memantau log secara langsung.

### 2. Menjalankan Secara Lokal (Langkah Manual)

Jika ingin menjalankan tanpa Docker, siapkan virtual environment terlebih dahulu:

```bash
# 1. Install dependencies menggunakan UV
uv sync

# 2. Jalankan MLflow Server lokal
mlflow server --host 127.0.0.1 --port 5000

# 3. Jalankan FL Server (Gunakan Terminal Baru)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m server.server

# 4. Jalankan FL Client (Gunakan Terminal Baru)
export SERVER_ADDRESS=127.0.0.1:8080
export DATASET_PATH=data/processed/heart_disease_cleaned.csv
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m client.client
```

---

## 👤 Author

* **Muhammad Attan** - [2022610013.attan@itp.ac.id](mailto:2022610013.attan@itp.ac.id)

---

## 📄 License

Proyek ini dilisensikan di bawah **MIT License** - lihat berkas [LICENSE](LICENSE) untuk detailnya.
