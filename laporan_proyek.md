# Laporan Proyek Machine Learning - [Nama Anda]

## Proyek Pertama: Predictive Analytics

### Domain Proyek

Penyakit kardiovaskular (CVDs) adalah penyebab utama kematian secara global, merenggut sekitar 17,9 juta nyawa setiap tahunnya. Deteksi dini dan diagnosis yang akurat memegang peranan krusial dalam manajemen penyakit jantung, yang dapat secara signifikan meningkatkan peluang keberhasilan pengobatan dan menyelamatkan nyawa. Latar belakang inilah yang mendorong pengembangan proyek _predictive analytics_ ini, yaitu untuk membangun sebuah model machine learning yang mampu memprediksi keberadaan penyakit jantung pada seorang pasien berdasarkan serangkaian data medis dan demografis.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan?**
Secara tradisional, diagnosis penyakit jantung memerlukan serangkaian tes yang kompleks, mahal, dan memakan waktu. Terdapat kebutuhan mendesak untuk alat bantu diagnosis yang lebih cepat, efisien, dan dapat diakses. Model machine learning dapat menganalisis pola tersembunyi dari data pasien yang mungkin sulit diidentifikasi oleh manusia. Dengan adanya model prediksi yang andal, tenaga medis dapat:

1.  Melakukan skrining awal dengan lebih cepat untuk mengidentifikasi pasien berisiko tinggi.
2.  Memprioritaskan pasien yang memerlukan perhatian medis lebih lanjut.
3.  Membantu dalam pengambilan keputusan klinis sebagai alat bantu diagnosis (bukan pengganti).

Solusi ini dapat mengurangi beban sistem kesehatan, menekan biaya diagnosis, dan yang terpenting, mempercepat intervensi medis bagi mereka yang paling membutuhkan.

**Referensi:**

1.  World Health Organization (WHO). (2021, June 11). _Cardiovascular diseases (CVDs)_. WHO. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
2.  Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). _Heart Disease Data Set_. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

### Business Understanding

#### Problem Statements

Berdasarkan latar belakang yang telah diuraikan, masalah yang ingin diselesaikan adalah:

- Bagaimana cara membangun sebuah model machine learning yang dapat memprediksi secara akurat apakah seorang pasien menderita penyakit jantung atau tidak, berdasarkan data klinis yang tersedia?
- Faktor atau fitur medis apa saja yang paling berpengaruh dalam memprediksi risiko penyakit jantung pada pasien?

#### Goals

Tujuan dari proyek ini adalah:

- Membuat model klasifikasi yang mampu memprediksi keberadaan penyakit jantung dengan metrik F1-Score di atas 85%, untuk memastikan keseimbangan antara presisi dan recall dalam konteks medis yang kritis.
- Mengidentifikasi fitur-fitur yang paling signifikan (paling prediktif) dari model terbaik yang dibangun, untuk memberikan wawasan tambahan bagi para profesional medis.

#### Solution Statements

Untuk mencapai tujuan tersebut, akan diajukan dua solusi machine learning sebagai berikut:

1.  **Membangun model klasifikasi menggunakan algoritma Logistic Regression.** Algoritma ini dipilih sebagai _baseline_ karena sifatnya yang sederhana, cepat, dan hasilnya mudah diinterpretasikan.
2.  **Membangun model klasifikasi menggunakan algoritma Random Forest.** Algoritma ini merupakan model _ensemble_ yang lebih kompleks dan kuat, yang diharapkan dapat memberikan performa lebih tinggi dibandingkan _baseline_ dengan menangkap pola non-linear dalam data.

Kedua model akan dievaluasi performanya menggunakan metrik Akurasi, Presisi, Recall, dan F1-Score pada data uji. Model dengan F1-Score tertinggi akan dipilih sebagai solusi terbaik.

---

### Data Understanding

Dataset yang digunakan adalah "Heart Disease UCI", yang bersumber dari Cleveland Clinic Foundation. Dataset ini dapat diakses melalui UCI Machine Learning Repository.

**Sumber Dataset:** [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

Dataset ini terdiri dari 303 sampel (baris) dan 14 fitur (kolom). Tidak terdapat nilai yang hilang (_missing values_) pada dataset ini setelah dilakukan pemeriksaan awal. Target variabel adalah 'target', yang bernilai 1 jika pasien memiliki penyakit jantung dan 0 jika tidak.

#### Variabel-variabel pada Dataset:

1.  **age**: Usia pasien (tahun).
2.  **sex**: Jenis kelamin (1 = pria; 0 = wanita).
3.  **cp**: Tipe nyeri dada (0-3).
4.  **trestbps**: Tekanan darah saat istirahat (mm Hg).
5.  **chol**: Kolesterol serum (mg/dl).
6.  **fbs**: Gula darah puasa > 120 mg/dl (1 = ya; 0 = tidak).
7.  **restecg**: Hasil elektrokardiografi istirahat (0-2).
8.  **thalach**: Denyut jantung maksimum yang dicapai.
9.  **exang**: Angina akibat olahraga (1 = ya; 0 = tidak).
10. **oldpeak**: Depresi ST yang diinduksi oleh latihan relatif terhadap istirahat.
11. **slope**: Kemiringan segmen ST latihan puncak.
12. **ca**: Jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluoroskopi.
13. **thal**: Status thalasemia (1 = normal; 2 = cacat tetap; 3 = cacat reversibel).
14. **target**: Diagnosis penyakit jantung (0 = tidak; 1 = ya).

#### Exploratory Data Analysis (EDA)

Tahapan EDA dilakukan untuk memahami data lebih dalam. Beberapa temuan penting:

- **Distribusi Target:** Terdapat 165 pasien (54.5%) yang didiagnosis menderita penyakit jantung dan 138 pasien (45.5%) yang tidak. Distribusi kelas ini cukup seimbang.
- **Distribusi Fitur:** Fitur numerik seperti `age`, `trestbps`, dan `chol` menunjukkan distribusi yang mendekati normal.
- **Korelasi Fitur:** Ditemukan korelasi positif antara `cp` (tipe nyeri dada) dan `thalach` (denyut jantung maks) dengan variabel `target`. Sebaliknya, `exang` (angina akibat olahraga) dan `oldpeak` memiliki korelasi negatif yang kuat dengan `target`.

![Distribusi Variabel Target](https://i.imgur.com/your-image-url1.png) ![Heatmap Korelasi](https://i.imgur.com/your-image-url2.png) ---

### Data Preparation

Proses persiapan data dilakukan untuk memastikan data siap digunakan untuk pemodelan.

1.  **Pemisahan Data (Train-Test Split):** Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan fungsi `train_test_split` dari Scikit-learn. Ini penting untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya. Stratifikasi (`stratify=y`) digunakan untuk memastikan proporsi kelas target pada data latih dan uji sama dengan proporsi pada dataset asli.
2.  **Standarisasi Fitur (Feature Scaling):** Fitur-fitur numerik (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) diskalakan menggunakan `StandardScaler`.
    - **Alasan:** Teknik ini diperlukan karena algoritma seperti Logistic Regression sensitif terhadap skala nilai fitur. Tanpa standarisasi, fitur dengan rentang nilai yang lebih besar dapat mendominasi proses pembelajaran model. Standarisasi mengubah fitur sehingga memiliki rata-rata 0 dan standar deviasi 1, memastikan semua fitur diperlakukan setara.

---

### Modeling

Dua model machine learning dibangun sesuai dengan _solution statement_.

1.  **Logistic Regression**:

    - Model ini dilatih pada data latih yang telah distandarisasi. Parameter yang digunakan adalah parameter _default_ dari Scikit-learn, yang sudah teroptimasi dengan baik untuk kasus umum.
    - **Kelebihan**: Cepat, sederhana, dan koefisiennya dapat diinterpretasikan untuk memahami pengaruh setiap fitur.
    - **Kekurangan**: Cenderung kurang berkinerja baik pada masalah dengan hubungan non-linear yang kompleks antara fitur dan target.

2.  **Random Forest**:
    - Model ini juga dilatih pada data latih yang sama. Parameter yang digunakan adalah `n_estimators=100`, `max_depth=10`, dan `random_state=42` untuk memastikan hasil yang dapat direproduksi.
    - **Kelebihan**: Kemampuan tinggi untuk menangkap pola kompleks dan non-linear, serta lebih tahan terhadap _overfitting_ dibandingkan satu _decision tree_.
    - **Kekurangan**: Cenderung menjadi "kotak hitam" (_black box_) karena hasilnya lebih sulit diinterpretasikan, dan memerlukan lebih banyak sumber daya komputasi.

**Pemilihan Model Terbaik:**
Berdasarkan hasil evaluasi, **Random Forest dipilih sebagai model terbaik**. Alasan utamanya adalah karena model ini memberikan F1-Score yang lebih tinggi pada data uji. Dalam konteks medis, F1-Score sangat penting karena menyeimbangkan antara Presisi (tidak salah mendiagnosis pasien sehat) dan Recall (tidak melewatkan pasien yang sakit), yang keduanya sama-sama krusial.

---

### Evaluation

Evaluasi model dilakukan pada data uji (20% dari dataset) untuk mengukur seberapa baik model dapat digeneralisasi pada data baru.

#### Metrik Evaluasi

Metrik yang digunakan untuk mengevaluasi performa model klasifikasi adalah:

- **Akurasi**: Rasio prediksi yang benar terhadap total data.
  - Formula: $Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$
  - Akurasi baik digunakan saat dataset seimbang, seperti dalam kasus ini.
- **Presisi**: Dari semua yang diprediksi positif, berapa persen yang sebenarnya positif.
  - Formula: $Precision = \frac{TP}{TP+FP}$
  - Penting untuk meminimalkan _False Positive_ (misal: tidak ingin menyebabkan kepanikan pada pasien sehat).
- **Recall (Sensitivitas)**: Dari semua yang sebenarnya positif, berapa persen yang berhasil diprediksi positif.
  - Formula: $Recall = \frac{TP}{TP+FN}$
  - Sangat penting dalam medis untuk meminimalkan _False Negative_ (tidak ingin melewatkan pasien yang sakit).
- **F1-Score**: Rata-rata harmonik dari Presisi dan Recall.
  - Formula: $F1 Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
  - Metrik ini memberikan gambaran yang seimbang tentang kinerja model, terutama ketika ada konsekuensi serius dari _False Positive_ dan _False Negative_.

#### Hasil Evaluasi

Berikut adalah perbandingan hasil evaluasi kedua model pada data uji:

| Metrik   | Logistic Regression | Random Forest |
| -------- | ------------------- | ------------- |
| Akurasi  | 0.85                | **0.87**      |
| Presisi  | 0.84                | **0.88**      |
| Recall   | 0.91                | 0.91          |
| F1-Score | 0.87                | **0.89**      |

Dari tabel di atas, terlihat bahwa model **Random Forest** unggul di hampir semua metrik, terutama pada **Akurasi, Presisi, dan F1-Score**. Meskipun memiliki nilai Recall yang sama, F1-Score yang lebih tinggi menunjukkan bahwa Random Forest memberikan keseimbangan terbaik antara Presisi dan Recall, menjadikannya model pilihan untuk solusi masalah ini.
