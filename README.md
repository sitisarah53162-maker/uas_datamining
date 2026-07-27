# UAS Data Mining - Implementasi Supervised & Unsupervised Learning

**Nama:** Siti Sarah  
**NIM:** 23146003  
**Mata Kuliah:** Data Mining
**Dosen Pengampu:** Teuku Rizky Noviandy, S.Kom., M.Kom.  

🔗 **Link Repositori GitHub:** [https://github.com/sitisarah53162-maker/uas_datamining](https://github.com/sitisarah53162-maker/uas_datamining)  
🌐 **Link Aplikasi Live (Streamlit):** [https://uasdatamining-sitisarah.streamlit.app](https://uasdatamining-sitisarah.streamlit.app) 

---

## 📌 Deskripsi Proyek
Aplikasi berbasis web ini dikembangkan menggunakan **Streamlit** untuk mendemonstrasikan dua metode utama dalam Data Mining:

1. **Klasifikasi Diabetes (Supervised Learning):**
   * Memprediksi risiko pasien mengidap diabetes berdasarkan indikator kesehatan medis.
   * Menggunakan dan membandingkan 3 algoritma: **K-Nearest Neighbors (KNN)**, **Naïve Bayes**, dan **Decision Tree**.
   * Menampilkan evaluasi kinerja model (Akurasi, Precision, Recall, F1-Score) dan matriks evaluasi (*Confusion Matrix*).

2. **Clustering Gerai Kopi & Deteksi Zona Sepi (Unsupervised Learning):**
   * Mengelompokkan titik lokasi gerai kopi menggunakan algoritma **K-Means Clustering** ($K = 3$).
   * Menganalisis parameter lingkungan (koordinat, kepadatan penduduk, arus lalu lintas, jumlah kompetitor, dan status area komersial).
   * Mengidentifikasi klaster yang dikategorikan sebagai **Zona Sepi Pelanggan** berdasarkan rata-rata tingkat keramaian terendah.

---

## 📁 Struktur File Proyek
```text
UAS_DATAMINING/
│
├── .pkl files                    # File model & scaler yang sudah dilatih
├── gerai_kopi_clustered.csv     # Dataset gerai kopi hasil clustering
├── diabetes.csv                 # Dataset medis diabetes
├── lokasi_gerai_kopi_clean.csv  # Dataset awal lokasi gerai kopi
├── train_model.py               # Script untuk melatih & menyimpan model
├── app.py                       # Aplikasi web utama Streamlit
├── requirements.txt             # Daftar pustaka dependency Python
└── README.md                    # Dokumentasi proyek
