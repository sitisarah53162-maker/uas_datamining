import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="UAS Data Mining - Streamlit App",
    page_icon="📊",
    layout="wide"
)

# Navigation Sidebar
st.sidebar.title("📌 Navigasi Menu")
menu = st.sidebar.radio("Pilih Topik Data Mining:", ["Klasifikasi Diabetes", "Clustering Gerai Kopi"])

# ==============================================================================
# MENU 1: KLASIFIKASI DIABETES
# ==============================================================================
if menu == "Klasifikasi Diabetes":
    st.title("🩺 Prediksi Risiko Diabetes Berdasarkan Data Pasien")
    st.markdown("""
    Aplikasi ini menggunakan metode **Supervised Learning** untuk memprediksi apakah seorang pasien berpotensi mengidap diabetes berdasarkan indikator medis.
    Tiga algoritma yang dievaluasi adalah **K-Nearest Neighbors (KNN)**, **Naïve Bayes**, dan **Decision Tree**.
    """)
    st.write("---")

    @st.cache_resource
    def load_classification_models():
        scaler = pickle.load(open('scaler_diabetes.pkl', 'rb'))
        knn = pickle.load(open('model_knn.pkl', 'rb'))
        nb = pickle.load(open('model_naive_bayes.pkl', 'rb'))
        dt = pickle.load(open('model_decision_tree.pkl', 'rb'))
        return scaler, knn, nb, dt

    try:
        scaler_diab, model_knn, model_nb, model_dt = load_classification_models()
        df_diab = pd.read_csv('diabetes.csv')
        
        # Split Data untuk Evaluasi Model
        X_diab = df_diab.drop('Outcome', axis=1)
        y_diab = df_diab['Outcome']
        _, X_test, _, y_test = train_test_split(X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab)
        X_test_scaled = scaler_diab.transform(X_test)

        models_dict = {
            "K-Nearest Neighbors (KNN)": model_knn,
            "Naïve Bayes": model_nb,
            "Decision Tree": model_dt
        }

        # ----------------------------------------------------------------------
        # 1. TABEL EVALUASI PERBANDINGAN SEMUA MODEL
        # ----------------------------------------------------------------------
        st.subheader("📋 Tabel Evaluasi Metrik Performa Model")
        
        metrics_list = []
        for name, model in models_dict.items():
            preds = model.predict(X_test_scaled)
            metrics_list.append({
                "Model Algoritma": name,
                "Akurasi": f"{accuracy_score(y_test, preds):.2%}",
                "Precision": f"{precision_score(y_test, preds):.2%}",
                "Recall": f"{recall_score(y_test, preds):.2%}",
                "F1-Score": f"{f1_score(y_test, preds):.2%}"
            })
        
        df_summary = pd.DataFrame(metrics_list)
        # Menampilkan tabel interaktif satu tampilan utuh
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.write("---")

        # ----------------------------------------------------------------------
        # 2. CONFUSION MATRIX & FORM PREDIKSI PASIEN BARU
        # ----------------------------------------------------------------------
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Visualisasi Confusion Matrix")
            
            selected_eval_model = st.selectbox("Pilih Model untuk Dilihat Confusion Matrix-nya:", list(models_dict.keys()))
            chosen_model = models_dict[selected_eval_model]
            
            y_pred = chosen_model.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Negatif (0)', 'Positif (1)'],
                        yticklabels=['Negatif (0)', 'Positif (1)'])
            ax.set_xlabel('Prediksi Model')
            ax.set_ylabel('Aktual')
            ax.set_title(f'Confusion Matrix - {selected_eval_model}')
            st.pyplot(fig)

        with col2:
            st.subheader("🔮 Form Prediksi Pasien Baru")
            st.write("Isi variabel medis pasien di bawah ini:")
            
            pregnancies = st.number_input("Pregnancies (Jumlah Kehamilan)", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose (Kadar Glukosa)", min_value=0, max_value=300, value=120)
            bp = st.number_input("Blood Pressure (Tekanan Darah)", min_value=0, max_value=200, value=70)
            skin = st.number_input("Skin Thickness (Ketebalan Kulit)", min_value=0, max_value=100, value=20)
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age (Usia)", min_value=1, max_value=120, value=30)
            
            model_choice = st.selectbox("Pilih Model Algoritma untuk Prediksi:", list(models_dict.keys()))

            if st.button("Jalankan Prediksi Risiko", type="primary"):
                input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                input_scaled = scaler_diab.transform(input_data)
                
                selected_model = models_dict[model_choice]
                prediction = selected_model.predict(input_scaled)[0]
                
                st.write("---")
                if prediction == 1:
                    st.error("⚠️ **HASIL PREDIKSI: POSITIF DIABETES**\n\nPasien terindikasi memiliki risiko diabetes tinggi.")
                else:
                    st.success("✅ **HASIL PREDIKSI: NEGATIF DIABETES**\n\nPasien terindikasi berada dalam batas normal / risiko rendah.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: File .pkl / dataset klasifikasi belum siap. Jalankan 'train_model.py' terlebih dahulu! ({e})")

# ==============================================================================
# MENU 2: CLUSTERING GERAI KOPI
# ==============================================================================
elif menu == "Clustering Gerai Kopi":
    st.title("☕ Analisis Klaster Lokasi Gerai Kopi & Deteksi Zona Sepi")
    st.markdown("""
    Aplikasi ini menggunakan metode **Unsupervised Learning (K-Means Clustering)** untuk mengelompokkan lokasi gerai kopi 
    berdasarkan koordinat dan parameter lingkungan guna mengidentifikasi **Zona Sepi Pelanggan**.
    """)
    st.write("---")

    @st.cache_resource
    def load_clustering_models():
        scaler = pickle.load(open('scaler_kopi.pkl', 'rb'))
        kmeans = pickle.load(open('model_kmeans_kopi.pkl', 'rb'))
        zona_sepi_cluster = pickle.load(open('zona_sepi_cluster.pkl', 'rb'))
        return scaler, kmeans, zona_sepi_cluster

    try:
        scaler_kopi, kmeans_kopi, zona_sepi_cluster = load_clustering_models()
        df_kopi = pd.read_csv('gerai_kopi_clustered.csv')

        col1, col2 = st.columns([1.2, 0.8])

        with col1:
            st.subheader("🗺️ Visualisasi Peta Spasial Gerai Kopi")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            clusters = df_kopi['cluster'].unique()
            colors = {0: 'blue', 1: 'red', 2: 'green'}
            labels = {
                zona_sepi_cluster: f"Cluster {zona_sepi_cluster} (ZONA SEPI)",
            }
            
            for c in clusters:
                label_text = labels.get(c, f"Cluster {c} (Zona Ramai/Komersial)")
                sub = df_kopi[df_kopi['cluster'] == c]
                ax.scatter(sub['x'], sub['y'], c=colors.get(c, 'gray'), label=label_text, alpha=0.6, edgecolors='w', s=50)

            ax.set_xlabel("Koordinat X")
            ax.set_ylabel("Koordinat Y")
            ax.set_title("Sebaran Lokasi Gerai Kopi berdasarkan Klaster")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        with col2:
            st.subheader("📍 Cek Klaster Lokasi Baru")
            st.write("Masukkan parameter lokasi gerai kopi baru:")
            
            input_x = st.number_input("Koordinat X", value=50.0)
            input_y = st.number_input("Koordinat Y", value=50.0)
            input_pop = st.number_input("Kepadatan Penduduk (Population Density)", min_value=0.0, value=1500.0)
            input_traffic = st.number_input("Arus Lalu Lintas (Traffic Flow)", min_value=0.0, value=600.0)
            input_comp = st.number_input("Jumlah Kompetitor", min_value=0, value=1)
            input_comm = st.selectbox("Area Komersial?", options=[(1, "Ya"), (0, "Tidak")], format_func=lambda x: x[1])[0]

            if st.button("Analisis Lokasi Baru", type="primary"):
                new_data = np.array([[input_x, input_y, input_pop, input_traffic, input_comp, input_comm]])
                new_data_scaled = scaler_kopi.transform(new_data)
                
                predicted_cluster = kmeans_kopi.predict(new_data_scaled)[0]
                
                st.write("---")
                st.info(f"📍 **Lokasi baru masuk ke dalam: Cluster {predicted_cluster}**")
                
                if predicted_cluster == zona_sepi_cluster:
                    st.warning("⚠️ **KATEGORI ZONA SEPI!**\n\nLokasi ini berada pada area dengan kepadatan penduduk / lalu lintas rendah. Potensi jumlah pengunjung berisiko sepi.")
                else:
                    st.success("🎉 **KATEGORI ZONA RAMAI / POTENSIAL!**\n\nLokasi ini berada pada area yang strategis dengan arus lalu lintas tinggi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: File .pkl / dataset gerai kopi belum siap. Jalankan 'train_model.py' terlebih dahulu! ({e})")