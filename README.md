# Experiment Tracking and Model Registry on Heart Disease Classification using MLflow and Flower

Proyek ini merupakan implementasi Federated Learning (FL) untuk klasifikasi penyakit jantung menggunakan dataset UCI Heart Disease. Sistem ini menggunakan framework **Flower** untuk orkestrasi pembelajaran terfederasi dan **MLflow** untuk pelacakan eksperimen (experiment tracking) serta manajemen model (model registry).

## 🚀 Fitur Utama

- **Federated Learning**: Melatih model machine learning lintas perangkat tanpa saling membagikan data mentah menggunakan framework Flower.
- **Experiment Tracking**: Mencatat parameter, metrik, dan artifact setiap ronde pelatihan ke MLflow.
- **Model Registry**: Menyimpan dan mengelola versi model terbaik untuk deployment.
- **Dockerized**: Seluruh komponen (Server, Client, MLflow) dapat dijalankan dengan mudah menggunakan Docker Compose.

## 📊 Dataset

UCI Heart Disease Dataset: [Kaggle Link](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
Dataset dibagi menjadi beberapa bagian untuk mensimulasikan lingkungan multi-client (misal: data Hungaria, Swiss, Cleveland, dsb).

## 📂 Struktur Proyek

```text
.
├── configs/            # Konfigurasi global (MLflow, Flower, dsb)
├── data/               # Dataset mentah dan hasil pemrosesan
│   ├── raw/
│   └── processed/
├── src/                # Source code utama
│   ├── client/         # Implementasi Flower Client
│   ├── server/         # Implementasi Flower Server/Strategy
│   ├── models/         # Arsitektur model (Scikit-learn)
│   └── utils/          # Fungsi pembantu (data loading, preprocessing)
├── Dockerfile.client   # Build image untuk client
├── Dockerfile.server   # Build image untuk server
├── docker-compose.yml  # Orkestrasi seluruh layanan
└── pyproject.toml      # Dependency management (UV)
```

## 🛠️ Prasyarat

Sebelum memulai, pastikan Anda telah menginstal:
- [Python 3.11+](https://www.python.org/)
- [UV](https://github.com/astral-sh/uv) (Pengelola paket Python yang cepat)
- [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)

## ⚙️ Cara Menjalankan

### 1. Menggunakan Docker Compose (Direkomendasikan)

Ini adalah cara tercepat untuk menjalankan seluruh ekosistem (MLflow + FL Server + 2 Clients):

```bash
docker-compose up --build
```

Setelah berjalan:
- **MLflow Dashboard**: Buka [http://localhost:5000](http://localhost:5000)
- **Flower Server**: Berjalan di port `8080`

### 2. Menjalankan Secara Lokal (Manual)

Jika ingin menjalankan tanpa Docker, buat virtual environment terlebih dahulu:

```bash
# Install dependencies
uv sync

# 1. Jalankan MLflow Server
mlflow server --host 127.0.0.1 --port 5000

# 2. Jalankan FL Server (Terminal Baru)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m server.server

# 3. Jalankan FL Client (Terminal Baru)
export SERVER_ADDRESS=127.0.0.1:8080
export DATASET_PATH=data/processed/heart_disease_cleaned.csv
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -m client.client
```

## 👤 Author

- **Muhammad Attan** - [2022610013.attan@itp.ac.id](mailto:2022610013.attan@itp.ac.id)

## 📄 License

Proyek ini dilisensikan di bawah MIT License - lihat berkas [LICENSE](LICENSE) untuk detailnya.
