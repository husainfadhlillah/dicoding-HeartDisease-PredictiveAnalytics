# Proyek Predictive Analytics: Prediksi Penyakit Jantung 🩺

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/2560px-Pandas_logo.svg.png" width="100" alt="Pandas Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/2560px-NumPy_logo_2020.svg.png" width="100" alt="NumPy Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/1200px-Created_with_Matplotlib-logo.svg.png" width="60" alt="Matplotlib Logo">
  <img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" width="150" alt="Seaborn Logo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width="120" alt="Scikit-learn Logo">
</p>

Proyek ini bertujuan untuk membangun model _machine learning_ yang mampu memprediksi risiko penyakit jantung pada seorang pasien berdasarkan data klinis. Dengan pendekatan klasifikasi, proyek ini membandingkan performa model **Logistic Regression** dan **Random Forest** untuk menemukan solusi prediksi yang paling akurat dan andal.

---

## **Latar Belakang**

Penyakit kardiovaskular (CVDs) adalah penyebab utama kematian secara global, merenggut sekitar 17,9 juta nyawa setiap tahunnya. Deteksi dini dan diagnosis yang akurat memegang peranan krusial dalam manajemen penyakit jantung, karena dapat meningkatkan peluang keberhasilan pengobatan secara signifikan.

Secara tradisional, diagnosis penyakit jantung memerlukan serangkaian tes yang kompleks dan mahal. _Machine learning_ menawarkan solusi untuk menciptakan alat bantu diagnosis yang lebih cepat, efisien, dan dapat diakses, dengan menganalisis pola dari data pasien yang sudah ada untuk memprediksi risiko pada pasien baru.

---

## **Dataset**

Dataset yang digunakan dalam proyek ini adalah **"Heart Failure Prediction"** yang bersumber dari Kaggle. Dataset ini berisi 11 fitur klinis yang dapat digunakan untuk memprediksi kemungkinan penyakit jantung pada pasien.

- **`Age`**: Usia pasien [tahun].
- **`Sex`**: Jenis kelamin pasien [M: Pria, F: Wanita].
- **`ChestPainType`**: Tipe nyeri dada [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic].
- **`RestingBP`**: Tekanan darah saat istirahat [mm Hg].
- **`Cholesterol`**: Kadar kolesterol serum [mm/dl].
- **`FastingBS`**: Gula darah puasa [1: jika > 120 mg/dl, 0: sebaliknya].
- **`RestingECG`**: Hasil elektrokardiogram istirahat [Normal, ST, LVH].
- **`MaxHR`**: Detak jantung maksimum yang dicapai.
- **`ExerciseAngina`**: Angina akibat olahraga [Y: Ya, N: Tidak].
- **`Oldpeak`**: Depresi ST yang diinduksi oleh latihan relatif terhadap istirahat.
- **`ST_Slope`**: Kemiringan segmen ST latihan puncak [Up, Flat, Down].
- **`HeartDisease`**: Variabel target [1: Gagal jantung, 0: Normal].

---

## **Alur Kerja Proyek**

Proyek ini dilaksanakan melalui beberapa tahapan utama:

1.  **Data Loading**: Memuat dataset `heart.csv` menggunakan Pandas.
2.  **Exploratory Data Analysis (EDA)**: Menganalisis distribusi data, hubungan antar fitur, dan statistik deskriptif untuk mendapatkan wawasan awal. Tahap ini penting untuk memahami karakteristik data sebelum pemodelan.
3.  **Data Preparation & Preprocessing**:
    - Melakukan _one-hot encoding_ untuk mengubah fitur kategorikal menjadi format numerik.
    - Membagi dataset menjadi data latih (_training set_) dan data uji (_testing set_) dengan proporsi 80:20.
    - Menerapkan standardisasi (menggunakan `StandardScaler`) pada fitur numerik untuk menyamakan skala data.
4.  **Modeling**:
    - Membangun model klasifikasi menggunakan **Logistic Regression**.
    - Membangun model klasifikasi menggunakan **Random Forest**.
5.  **Hyperparameter Tuning**: Melakukan optimisasi pada model Random Forest menggunakan `GridSearchCV` untuk menemukan kombinasi parameter terbaik yang menghasilkan performa tertinggi.
6.  **Evaluasi Model**: Mengevaluasi kedua model menggunakan metrik **Akurasi, Presisi, Recall, dan F1-Score**. Analisis _Confusion Matrix_ juga dilakukan untuk memahami tipe kesalahan yang dibuat oleh model.

---

## **Hasil dan Evaluasi**

Setelah melalui tahapan pemodelan dan evaluasi, diperoleh perbandingan performa sebagai berikut:

| Model                     | Akurasi   | Presisi   | Recall    | F1-Score  |
| :------------------------ | :-------- | :-------- | :-------- | :-------- |
| Logistic Regression       | 0.886     | 0.872     | 0.931     | 0.900     |
| **Random Forest (Tuned)** | **0.902** | **0.889** | **0.941** | **0.914** |

<br>

**Analisis Hasil:**

- **Performa Unggul**: Model **Random Forest yang telah dioptimalkan (Tuned)** menunjukkan performa yang lebih unggul di semua metrik evaluasi dibandingkan dengan Logistic Regression.
- **Mengurangi Risiko Fatal**: Dalam konteks medis, metrik **Recall** dan **meminimalkan False Negative** (pasien sakit diprediksi sehat) sangat krusial. Model Random Forest berhasil mengurangi jumlah _False Negative_ menjadi hanya 6 kasus, lebih baik dari Logistic Regression (7 kasus).

**Kesimpulan:**
Berdasarkan performa metrik dan analisis risiko, **Random Forest** terpilih sebagai model terbaik untuk proyek ini karena memberikan akurasi tertinggi dan lebih andal dalam mendeteksi pasien yang benar-benar berisiko terkena penyakit jantung.

---

## **Struktur Repository**

```
├── heart.csv
├── notebook_proyek.ipynb
├── laporan_proyek.md
└── README.md
```

- **`heart.csv`**: Dataset mentah yang digunakan.
- **`notebook_proyek.ipynb`**: File Jupyter Notebook yang berisi seluruh kode analisis dan pemodelan.
- **`laporan_proyek.md`**: Laporan proyek dalam format markdown.
- **`README.md`**: Penjelasan mengenai proyek ini.

---

## **Cara Menjalankan Proyek**

Untuk menjalankan notebook proyek ini di lingkungan lokal, ikuti langkah-langkah berikut:

1.  **Clone Repository**

    ```bash
    git clone [https://github.com/NAMA_USER/NAMA_REPO.git](https://github.com/NAMA_USER/NAMA_REPO.git)
    cd NAMA_REPO
    ```

2.  **Buat Lingkungan Virtual (Opsional tapi Direkomendasikan)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instal Dependensi**
    Pastikan Anda memiliki file `requirements.txt` atau instal library yang dibutuhkan secara manual:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```

4.  **Jalankan Jupyter Notebook**
    ```bash
    jupyter notebook notebook_proyek.ipynb
    ```

```

```
