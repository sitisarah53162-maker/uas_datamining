import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

print("==================================================")
print("1. PROSES KLASIFIKASI DIABETES (SUPERVISED LEARNING)")
print("==================================================")

try:
    # 1. Load Dataset Diabetes
    df_diabetes = pd.read_csv('diabetes.csv')

    X_diab = df_diabetes.drop('Outcome', axis=1)
    y_diab = df_diabetes['Outcome']

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab
    )

    # Standard Scaling
    scaler_diabetes = StandardScaler()
    X_train_scaled = scaler_diabetes.fit_transform(X_train)
    X_test_scaled = scaler_diabetes.transform(X_test)

    # Inisialisasi Model Klasifikasi
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    results = {}
    print("\nHasil Evaluasi Model Klasifikasi:")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"[{name}] Akurasi: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        # Simpan masing-masing model
        file_name = f"model_{name.lower().replace(' ', '_')}.pkl"
        pickle.dump(model, open(file_name, 'wb'))

    # Simpan Scaler Diabetes
    pickle.dump(scaler_diabetes, open('scaler_diabetes.pkl', 'wb'))
    print("-> Semua model klasifikasi & scaler diabetes berhasil disimpan!")

except FileNotFoundError:
    print("⚠️ File 'diabetes.csv' tidak ditemukan. Pastikan file ada di folder proyek.")

print("\n==================================================")
print("2. PROSES CLUSTERING GERAI KOPI (UNSUPERVISED LEARNING)")
print("==================================================")

try:
    # 1. Load Dataset Gerai Kopi
    df_kopi = pd.read_csv('lokasi_gerai_kopi_clean.csv')

    features_kopi = ['x', 'y', 'population_density', 'traffic_flow', 'competitor_count', 'is_commercial']
    X_kopi = df_kopi[features_kopi]

    # Standard Scaling
    scaler_kopi = StandardScaler()
    X_kopi_scaled = scaler_kopi.fit_transform(X_kopi)

    # K-Means Clustering (K = 3)
    kmeans_kopi = KMeans(n_clusters=3, random_state=42)
    df_kopi['cluster'] = kmeans_kopi.fit_predict(X_kopi_scaled)

    # Identifikasi Zona Sepi berdasarkan rata-rata traffic & kepadatan terendah
    cluster_means = df_kopi.groupby('cluster')[['population_density', 'traffic_flow']].mean()
    zona_sepi_cluster = cluster_means['traffic_flow'].idxmin()

    print(f"\nRata-rata Parameter per Klaster:")
    print(cluster_means)
    print(f"\n-> Cluster {zona_sepi_cluster} teridentifikasi sebagai ZONA SEPI (Traffic & Kepadatan terendah).")

    # Simpan Model K-Means, Scaler, dan Label Zona Sepi
    pickle.dump(scaler_kopi, open('scaler_kopi.pkl', 'wb'))
    pickle.dump(kmeans_kopi, open('model_kmeans_kopi.pkl', 'wb'))
    pickle.dump(zona_sepi_cluster, open('zona_sepi_cluster.pkl', 'wb'))
    
    # Simpan dataset hasil clustering untuk visualisasi Streamlit
    df_kopi.to_csv('gerai_kopi_clustered.csv', index=False)
    print("-> Model K-Means, scaler, dan dataset clustered berhasil disimpan!")

except FileNotFoundError:
    print("⚠️ File 'lokasi_gerai_kopi_clean.csv' tidak ditemukan. Pastikan file ada di folder proyek.")

print("\n==================================================")
print("SELESAI! JALANKAN 'python -m streamlit run app.py' UNTUK MEMBUKA WEB.")
print("==================================================")