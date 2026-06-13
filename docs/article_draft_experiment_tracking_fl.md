# Implementasi *Experiment Tracking* dan *Model Registry* Terpusat pada Lingkungan *Federated Learning* Menggunakan MLflow dan Flower

**Penulis:** [Nama Penulis]
**Afiliasi:** [Nama Institusi]
**Email:** [email@institusi.ac.id]

---

## Abstrak

Data medis bersifat sangat sensitif dan seringkali tersebar di berbagai institusi layanan kesehatan yang tidak dapat berbagi data secara langsung akibat regulasi privasi yang ketat. *Federated Learning* (FL) hadir sebagai solusi pembelajaran mesin kolaboratif tanpa sentralisasi data mentah. Meskipun demikian, literatur FL saat ini masih didominasi oleh riset pada level algoritma agregasi, sementara tantangan operasional manajemen siklus hidup model (*MLOps*) pada lingkungan federasi belum banyak dibahas. Penelitian ini merancang dan mengimplementasikan arsitektur *experiment tracking* dan *model registry* terpusat yang mengintegrasikan framework federasi **Flower** dengan platform MLOps **MLflow**. Kontribusi utama meliputi: (1) mekanisme pemetaan hierarki *Parent-Child Run* antara server FL dan client FL; (2) pencatatan metrik operasional federasi berbasis *weighted-average* termasuk pendeteksian bottleneck komputasi (*straggler*); dan (3) otomatisasi seleksi dan promosi versi model global terbaik ke *Model Registry* produksi. Sistem ini diujikan pada studi kasus klasifikasi penyakit jantung menggunakan dataset UCI Heart Disease yang didistribusikan secara *non-IID* ke dua client node. Hasil eksperimen menunjukkan bahwa arsitektur yang diusulkan mampu memberikan visibilitas penuh terhadap proses pelatihan terdistribusi tanpa melanggar prinsip privasi *Federated Learning*.

**Kata kunci:** *Federated Learning*, *Experiment Tracking*, *Model Registry*, MLflow, Flower, MLOps, Klasifikasi Penyakit Jantung

---

## 1. Pendahuluan

Penyakit jantung merupakan salah satu penyebab kematian utama di dunia, bertanggung jawab atas sekitar 17,9 juta kematian per tahun menurut laporan World Health Organization [1]. Kemajuan *machine learning* (ML) menawarkan potensi besar dalam membantu diagnosis dini penyakit ini melalui pembelajaran pola pada data klinis. Namun, pengembangan model ML yang efektif membutuhkan volume data pelatihan yang besar dan beragam, yang dalam konteks medis tersimpan secara tersebar di berbagai rumah sakit dan klinik yang tidak dapat saling berbagi data secara langsung.

Hambatan utama pengumpulan data medis berskala besar adalah regulasi privasi yang ketat, seperti *Health Insurance Portability and Accountability Act* (HIPAA) di Amerika Serikat dan *General Data Protection Regulation* (GDPR) di Uni Eropa [2]. Regulasi tersebut melarang pemindahan data rekam medis pasien ke pihak ketiga tanpa persetujuan eksplisit, sehingga menciptakan "silo data" antar institusi [3].

*Federated Learning* (FL) diusulkan oleh McMahan et al. [4] sebagai paradigma pembelajaran mesin kolaboratif yang memungkinkan model dilatih dari data yang terdistribusi di berbagai node lokal (*client*) tanpa perlu memindahkan data mentah ke server pusat. Algoritma *Federated Averaging* (FedAvg) yang diperkenalkan pada studi tersebut menjadi fondasi dari sebagian besar penelitian FL hingga saat ini.

Meskipun FL telah terbukti efektif dalam berbagai studi kasus klinis [3][5], mayoritas penelitian FL masih berfokus pada inovasi di level algoritma agregasi dan analisis konvergensi teoritis [6][7]. Terdapat kesenjangan penelitian yang signifikan terkait aspek rekayasa perangkat lunak dan operasional (*MLOps*) pada lingkungan federasi, khususnya:

1. **Pelacakan Eksperimen Terdistribusi (*Experiment Tracking*):** Melacak konfigurasi hiperparameter, riwayat pelatihan, dan metrik performa dari banyak client yang berjalan secara independen dan asinkron sangat sulit dilakukan tanpa sistem pencatatan terpusat [8].
2. **Manajemen Siklus Hidup Model Global (*Model Lifecycle Management*):** Setelah pelatihan federasi selesai, proses rekonstruksi, validasi, dan penyimpanan model global terbaik ke dalam sistem siap-pakai (*production-ready*) sering kali dilakukan secara manual dan rentan terhadap kesalahan reprodusibilitas [9].

Penelitian ini bertujuan untuk mengisi kesenjangan tersebut dengan merancang dan mengimplementasikan sistem MLOps terintegrasi pada lingkungan FL. Sistem yang dibangun menggunakan framework **Flower** [10] sebagai mesin federasi dan platform **MLflow** [9] sebagai alat pelacakan eksperimen dan registri model. Implementasi divalidasi melalui studi kasus klasifikasi penyakit jantung menggunakan dataset UCI Heart Disease [11].

---

## 2. Tinjauan Pustaka

### 2.1 Federated Learning dan Algoritma FedAvg

McMahan et al. [4] mendefinisikan *Federated Learning* sebagai pendekatan pelatihan model ML di mana komputasi dilakukan secara lokal di node client dan hanya hasil pembaruan parameter (gradien atau bobot) yang dikirimkan ke server pusat untuk diagregasi. Pada setiap ronde komunikasi $t$, server mengirimkan bobot global $w_t$ ke $K$ client terpilih. Setiap client $k$ melakukan pembaruan lokal dengan mengoptimalkan loss lokal menggunakan data lokalnya, menghasilkan bobot lokal baru $w_{t+1}^k$. Server kemudian menghitung bobot global baru melalui rata-rata tertimbang (*FedAvg*):

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} w_{t+1}^k$$

di mana $n_k$ adalah jumlah sampel data client $k$ dan $N = \sum_{k=1}^K n_k$ adalah total sampel seluruh client yang berpartisipasi pada ronde tersebut.

Li et al. [6] mengidentifikasi heterogenitas data (*non-IID*) sebagai tantangan fundamental dalam FL, di mana distribusi data antar client yang tidak seragam dapat menyebabkan divergensi model lokal (*client drift*) yang memperlambat konvergensi global.

### 2.2 Federated Learning dalam Domain Medis

Rieke et al. [3] mengulas peran FL sebagai pendekatan yang menjanjikan untuk mengatasi "silo data" dalam kesehatan digital. Studi tersebut menegaskan bahwa FL memungkinkan kolaborasi antar institusi medis tanpa melanggar regulasi privasi, sekaligus mengidentifikasi tantangan kunci seperti heterogenitas statistik antar dataset rumah sakit dan kebutuhan akan standarisasi protokol evaluasi.

Aouedi et al. [5] secara khusus membahas tantangan keamanan dan privasi dalam penanganan data medis sensitif menggunakan FL, dan menyoroti bahwa mayoritas implementasi riset FL medis saat ini masih berupa simulasi berbasis dataset *benchmark* seperti UCI tanpa infrastruktur MLOps yang matang.

### 2.3 MLOps dan Tantangannya pada Lingkungan Federasi

MLflow, yang diperkenalkan oleh Zaharia et al. [9] dari Databricks pada 2018, merupakan platform *open-source* untuk manajemen siklus hidup ML yang mencakup pelacakan eksperimen, pengelolaan proyek, dan registri model. Zaharia et al. menyatakan bahwa platform MLOps seperti MLflow dirancang dengan antarmuka terbuka (*open-interface*) agar dapat bekerja dengan berbagai pustaka ML, namun arsitektur dasarnya masih mengasumsikan proses pelatihan yang terpusat.

Pendekatan *FLOps* yang berkembang dalam literatur terbaru [8] menunjukkan bahwa integrasi ML pipeline tradisional dengan lingkungan federasi memerlukan adaptasi signifikan, terutama dalam hal orkestrasi pencatatan metrik lintas node dan manajemen versi model global.

### 2.4 Framework Flower

Flower [10] adalah framework FL yang bersifat *framework-agnostic* dan mendukung berbagai tipe client (Python, PyTorch, TensorFlow) serta berbagai strategi agregasi. Arsitektur Flower yang berbasis callback strategy memudahkan ekstensi dan kustomisasi perilaku agregasi, menjadikannya pilihan yang tepat untuk integrasi dengan sistem pelacakan eksternal seperti MLflow.

---

## 3. Metodologi dan Arsitektur Sistem

Sistem yang dibangun mengintegrasikan siklus pelatihan terfederasi Flower dengan orkestrasi pencatatan MLflow melalui tiga mekanisme utama yang dijelaskan berikut.

### 3.1 Dataset dan Konfigurasi Eksperimen

Penelitian ini menggunakan *Heart Disease Dataset* dari UCI Machine Learning Repository [11], yang merupakan dataset biner untuk klasifikasi ada atau tidaknya penyakit jantung berdasarkan 13 atribut klinis pasien (usia, jenis kelamin, jenis nyeri dada, tekanan darah istirahat, kolesterol serum, dll.). Dataset asli terdiri dari 303 rekaman pasien. Untuk mensimulasikan skenario *Federated Learning*, dataset dibagi secara *non-IID* ke dua client node dengan rasio yang tidak seimbang untuk merepresentasikan kondisi distribusi data klinis nyata antar institusi.

Model klasifikasi yang digunakan adalah **Logistic Regression** dari pustaka scikit-learn, dengan konfigurasi solver SAGA yang mendukung berbagai jenis regularisasi (L1, L2, dan ElasticNet). Pemilihan model ini didasarkan pada interpretabilitasnya yang tinggi, yang relevan untuk konteks medis di mana penjelasan prediksi kepada klinisi menjadi kebutuhan penting.

Seluruh komponen sistem (server FL, client FL, server MLflow, dan monitoring stack) dikemas dalam container Docker menggunakan Docker Compose untuk menjamin reprodusibilitas dan kemudahan deployment.

### 3.2 Mekanisme Pemetaan Parent-Child Run di MLflow

Tantangan utama dalam pelacakan eksperimen FL adalah menyatukan sesi training dari proses-proses yang berjalan di mesin-mesin yang terpisah dalam satu hierarki eksperimen yang koheren. Penelitian ini menyelesaikan masalah ini melalui mekanisme *Parent-Child Run* sebagai berikut:

1. Saat proses federasi dimulai, **server FL** membuka sebuah *Parent Run* di MLflow dan mendapatkan `server_run_id` yang bersifat unik.
2. `server_run_id` ini disisipkan ke dalam kamus konfigurasi instruksi Flower melalui fungsi `on_fit_config_fn` dan `on_evaluate_config_fn`, yang kemudian dikirim ke seluruh client pada awal setiap ronde pelatihan.
3. Saat setiap **client FL** menerima instruksi training, client membuka sebuah *Child Run* dengan menyertakan tag `"mlflow.parentRunId": server_run_id`. Mekanisme ini memastikan seluruh metrik pelatihan lokal client tertata rapi dalam satu hierarki eksperimen yang sama pada dashboard MLflow, tanpa perlu koordinasi jaringan tambahan.

### 3.3 Kustomisasi Strategi Agregasi (MLflowStrategy)

Inti dari sistem pelacakan ini adalah kelas kustom `MLflowStrategy` yang memperluas (*extend*) kelas `fl.server.strategy.FedAvg` bawaan Flower. Strategi ini menambahkan orkestrasi pencatatan MLflow pada tiga titik penting dalam siklus federasi:

- **`aggregate_fit`:** Dipanggil setelah server menerima dan mengagregasi parameter model dari seluruh client. Metrik yang dicatat meliputi: rata-rata tertimbang akurasi training lokal (`fedavg_train_accuracy`), rata-rata tertimbang loss training lokal (`fedavg_train_loss`), rata-rata durasi komputasi lokal client (`fedavg_fit_duration`), dan durasi komputasi terlama di antara seluruh client (`max_fit_duration`) sebagai indikator *straggler*.
- **`aggregate_evaluate`:** Dipanggil setelah server menerima laporan evaluasi lokal dari client pada ronde akhir. Metrik yang dicatat meliputi akurasi evaluasi lokal rata-rata tertimbang (`fedavg_eval_accuracy`) dan loss evaluasi lokal (`fedavg_eval_loss`).
- **Evaluasi Terpusat (`evaluate`):** Fungsi terpisah yang dieksekusi oleh server menggunakan dataset uji lokal server setelah setiap ronde agregasi (kecuali ronde 0 untuk menghindari pencatatan model inisialisasi yang belum terlatih).

### 3.4 Otomatisasi Rekonstruksi dan Promosi Model Registry

Pada ronde federasi terakhir, sistem secara otomatis menjalankan alur berikut:

1. **Seleksi Model Terbaik:** Selama proses pelatihan, `MLflowStrategy` menyimpan salinan (*deep copy*) bobot parameter NDArray dari ronde dengan akurasi validasi tertinggi (`best_ndarrays`) beserta skor akurasinya (`best_accuracy`).
2. **Rekonstruksi Model:** Bobot terbaik tersebut dimasukkan kembali ke dalam sebuah instance model `LogisticRegression` baru untuk menghasilkan representasi model yang dapat diserialisasi.
3. **Registrasi ke MLflow:** Model diregistrasikan ke MLflow Model Registry menggunakan `mlflow.sklearn.log_model` dengan format serialisasi `skops` yang lebih aman dibandingkan *pickle* standar (meminimalisir risiko *pickle injection attack*).
4. **Promosi dan Pemberian Tag:** Menggunakan `MlflowClient`, sistem secara otomatis menandai versi model terbaru dengan tag performa (`validation_accuracy`, `best_round`) dan mempromosikannya dengan alias `@Production` agar siap digunakan oleh sistem hilir (*downstream*).

---

## 4. Hasil dan Analisis

### 4.1 Analisis Struktur Hierarki Eksperimen di MLflow

Hasil implementasi menunjukkan bahwa mekanisme *Parent-Child Run* berhasil memetakan seluruh aktivitas pelatihan lokal dari kedua client ke dalam satu hierarki yang terstruktur di bawah satu *parent run* server pada dashboard MLflow. Tanpa mekanisme ini, metrik dari setiap client akan tercatat sebagai *run* yang terisolasi dan terpisah, mempersulit analisis komparatif antar ronde.

### 4.2 Analisis Perbandingan Akurasi Global vs. Lokal (Utility Analysis)

Eksperimen dilakukan menggunakan konfigurasi 10 ronde federasi dengan 1–10 iterasi lokal per ronde. Performa model global diuji menggunakan dua strategi evaluasi yang saling melengkapi:

- **Evaluasi Terpusat:** Model global diuji pada dataset uji yang tersimpan di server, menghasilkan metrik `centralized_accuracy` yang konsisten dan tidak bias.
- **Evaluasi Terdistribusi (FedAvg):** Agregasi hasil evaluasi lokal client menghasilkan `fedavg_eval_accuracy` sebagai estimasi performa pada data lokal.

Hasil eksperimen menunjukkan bahwa nilai `centralized_accuracy` pada ronde akhir mencapai performa yang kompetitif dibandingkan *baseline* model terpusat konvensional yang dilatih dengan seluruh data terpusat, dengan penurunan (*utility drop*) yang sangat minimal.

### 4.3 Analisis Deteksi Straggler

Pemantauan metrik `fedavg_fit_duration` (rata-rata durasi training) dan `max_fit_duration` (durasi terlama) di dashboard MLflow memungkinkan identifikasi ketimpangan kapasitas komputasi antar client. Pada eksperimen dengan dua client dengan beban data berbeda, terpantau bahwa selisih antara `max_fit_duration` dan `fedavg_fit_duration` meningkat seiring dengan pertambahan iterasi lokal, mengindikasikan potensi bottleneck yang akan berdampak signifikan pada efisiensi komunikasi total jika jumlah client diperbesar.

### 4.4 Validasi Model Registry dan Siklus Deployment Otomatis

Setelah ronde federasi selesai, model dengan akurasi validasi tertinggi berhasil direkonstruksi, didaftarkan ke MLflow Model Registry, dan dipromosikan secara otomatis dengan alias `@Production` dalam satu alur kerja yang terintegrasi penuh. Hal ini memangkas proses deployment model dari operasi manual multi-langkah menjadi satu proses orkestrasi yang dapat direproduksi secara konsisten pada setiap eksperimen.

---

## 5. Diskusi

### 5.1 Dilema Strategi Evaluasi dalam Federated Learning

Pemilihan strategi evaluasi model global dalam FL merupakan isu fundamental yang memiliki implikasi langsung terhadap validitas hasil penelitian. Penelitian ini menganalisis dua pendekatan yang memiliki trade-off berbeda:

**Evaluasi Terpusat (*Centralized Evaluation*)** memberikan estimasi performa yang paling objektif karena model diuji pada satu set data uji yang konsisten, seimbang antar kelas, dan tidak terkontaminasi oleh distribusi data lokal client. Namun, dalam skenario medis nyata, mengumpulkan data pasien ke server pusat justru bertentangan dengan prinsip dasar FL. Pada penelitian ini, dataset UCI Heart Disease digunakan sebagai *proxy* untuk dataset uji server dalam konteks prototipe. Untuk penerapan klinis nyata, penggunaan **data sintetis berkualitas tinggi** yang dibangkitkan oleh model generatif (seperti *Conditional Tabular GAN*/CTGAN) merupakan alternatif yang menjanjikan. Pendekatan data sintetis memungkinkan server memiliki dataset uji yang merepresentasikan karakteristik statistik seluruh kelas klinis tanpa memaparkan data rekam medis pasien asli.

**Evaluasi Terdistribusi Lokal (*Client-Side Evaluation*)** sepenuhnya mematuhi prinsip privasi FL karena data uji tidak pernah meninggalkan node lokal client. Namun, pendekatan ini rentan terhadap **bias evaluasi lokal** yang disebabkan oleh distribusi data yang *non-IID* antar client. Pada skenario di mana data satu client didominasi oleh satu subkelompok populasi (misalnya: kelompok usia tertentu atau gender tertentu), metrik evaluasi lokal yang dihasilkan akan mencerminkan performa model pada subpopulasi tersebut, bukan pada populasi umum. Agregasi rata-rata tertimbang dari metrik yang bias ini menyulitkan penarikan kesimpulan tunggal yang valid mengenai performa model global secara keseluruhan.

### 5.2 Kontribusi terhadap MLOps Medis

Dibandingkan dengan penelitian FL yang pada umumnya hanya menyajikan hasil akhir berupa file bobot model dalam format NumPy array tanpa infrastruktur MLOps yang matang, arsitektur yang diusulkan dalam penelitian ini menyediakan: (a) jejak audit (*audit trail*) eksperimen yang lengkap dan dapat direproduksi; (b) mekanisme perbandingan performa model antar ronde yang terotomasi; dan (c) siklus deployment model yang aman dan terstandardisasi. Selain itu, penggunaan format serialisasi `skops` pada *Model Registry* memberikan lapisan keamanan tambahan dibandingkan format `pickle` standar yang rentan terhadap eksploitasi deserialisasi.

### 5.3 Keterbatasan Penelitian

Penelitian ini memiliki beberapa keterbatasan yang perlu diakui. Pertama, eksperimen dilakukan dengan hanya dua client node dalam lingkungan terkontrol (*containerized*), sehingga skalabilitas sistem belum diuji pada skenario dengan puluhan atau ratusan client. Kedua, dataset UCI Heart Disease yang digunakan merupakan dataset kecil (303 instans) yang tidak sepenuhnya merepresentasikan kompleksitas distribusi data klinis nyata dalam skala besar.

---

## 6. Kesimpulan

Penelitian ini berhasil merancang dan mengimplementasikan arsitektur MLOps terpusat yang secara efektif menjembatani kesenjangan antara proses pelatihan *Federated Learning* yang terdistribusi dengan kebutuhan manajemen eksperimen dan siklus hidup model yang terstruktur. Mekanisme *Parent-Child Run* yang diimplementasikan memungkinkan visibilitas penuh terhadap aktivitas training lintas node tanpa melanggar prinsip privasi data FL. Otomatisasi seleksi dan promosi model global terbaik ke tahap `@Production` memangkas proses deployment dari operasi manual menjadi alur yang terintegrasi dan dapat direproduksi.

Untuk penelitian selanjutnya, beberapa arah pengembangan yang direkomendasikan adalah: (1) pengujian skalabilitas sistem pada lingkungan dengan jumlah client yang lebih besar; (2) integrasi mekanisme pencatatan parameter privasi *Differential Privacy* (seperti budget epsilon $\varepsilon$) ke dalam dashboard MLflow sebagai metrik tambahan; dan (3) eksplorasi penggunaan data sintetis (CTGAN) sebagai pengganti dataset uji nyata pada server FL untuk menjaga konsistensi prinsip privasi pada semua komponen sistem.

---

## Daftar Pustaka

[1] World Health Organization. (2021). *Cardiovascular diseases (CVDs)*. WHO Fact Sheets. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

[2] Voigt, P., & Von dem Bussche, A. (2017). *The EU General Data Protection Regulation (GDPR): A Practical Guide*. Springer International Publishing. https://doi.org/10.1007/978-3-319-57959-7

[3] Rieke, N., Hancox, J., Li, W., Milletarì, F., Roth, H. R., Albarqouni, S., ... & Cardoso, M. J. (2020). The future of digital health with federated learning. *npj Digital Medicine*, *3*(1), 119. https://doi.org/10.1038/s41746-020-00323-1

[4] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54:1273–1282.

[5] Aouedi, O., Piamrat, K., & Muller, G. (2022). Handling privacy-sensitive medical data with federated learning: Challenges and future directions. *IEEE Journal of Biomedical and Health Informatics*, *27*(2), 790–803. https://doi.org/10.1109/JBHI.2022.3185673

[6] Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated learning: Challenges, methods, and future directions. *IEEE Signal Processing Magazine*, *37*(3), 50–60. https://doi.org/10.1109/MSP.2020.2975749

[7] Li, X., Huang, K., Yang, W., Wang, S., & Zhang, Z. (2020). On the convergence of FedAvg on Non-IID Data. In *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)*.

[8] Nepfl, J., & Torabi Rad, M. (2024). A survey on MLOps for federated learning: Challenges, practices, and future directions. arXiv preprint arXiv:2404.xxxxx. *(Placeholder – verifikasi nomor arXiv terkini).*

[9] Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., ... & Talwalkar, A. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Engineering Bulletin*, *41*(4), 39–45.

[10] Beutel, D. J., Topal, T., Mathur, A., Qiu, X., Fernandez-Marques, J., Gao, Y., ... & Lane, N. D. (2020). Flower: A friendly federated learning research framework. arXiv preprint arXiv:2007.14390.

[11] Detrano, R., Jánosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., ... & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, *64*(3), 304–310. https://doi.org/10.1016/0002-9149(89)90524-9

[12] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.
