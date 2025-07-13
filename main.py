import pandas as pd
import os
import glob
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np

# === Ayarlar ===
data_folder = 'datas'
txt_files = sorted(glob.glob(os.path.join(data_folder, '*.txt')))
batch_size = 10
model_path = 'kmeans_model.pkl'
scaler_path = 'scaler.pkl'
threshold_path = 'threshold.pkl'

# === Model ve scaler yükle veya eğit ===
if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(threshold_path):
    print("🔁 Model, scaler ve threshold bulundu, yükleniyor...")
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    threshold = joblib.load(threshold_path)
else:
    print("🚀 Eğitim başlatılıyor...")
    scaler = StandardScaler()
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=10000)

    all_distances = []

    for i in range(0, len(txt_files), batch_size):
        batch_files = txt_files[i:i + batch_size]
        df_list = []

        for file in batch_files:
            df = pd.read_csv(file, sep="\t", header=None)
            df.columns = ['user_id', 'timestamp', 'duration', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            df = df.rename(columns={
                'feature1': 'sms_volume',
                'feature2': 'call_volume',
                'feature3': 'internet_download',
                'feature4': 'internet_upload',
                'feature5': 'total_traffic_score'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['sms_volume', 'call_volume', 'internet_download', 'internet_upload', 'total_traffic_score']:
                df[col] = df[col].fillna(0)
            df_list.append(df)

        batch_df = pd.concat(df_list, ignore_index=True)
        features = ['duration', 'sms_volume', 'call_volume', 'internet_download', 'internet_upload']
        X_batch = batch_df[features]

        scaler.partial_fit(X_batch)
        X_scaled = scaler.transform(X_batch)

        kmeans.partial_fit(X_scaled)
        distances = cdist(X_scaled, kmeans.cluster_centers_, 'euclidean')
        all_distances.append(distances.min(axis=1))

        print(f"✅ Batch {i//batch_size+1} işlendi.")

    all_distances = np.concatenate(all_distances)
    threshold = np.quantile(all_distances, 0.95)
    print(f"📏 Threshold (0.95 quantile): {threshold:.2f}")

    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(threshold, threshold_path)
    print("✅ Model ve scaler kaydedildi.")


# === Test dosyası ===
test_file = os.path.join(data_folder, 'sms-call-internet-mi-2013-11-10.txt')
print(f"📂 Test verisi yükleniyor: {os.path.basename(test_file)}")
df = pd.read_csv(test_file, sep="\t", header=None)
df.columns = ['user_id', 'timestamp', 'duration', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5']
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.rename(columns={
    'feature1': 'sms_volume',
    'feature2': 'call_volume',
    'feature3': 'internet_download',
    'feature4': 'internet_upload',
    'feature5': 'total_traffic_score'
})
for col in ['sms_volume', 'call_volume', 'internet_download', 'internet_upload', 'total_traffic_score']:
    df[col] = df[col].fillna(0)

features = ['duration', 'sms_volume', 'call_volume', 'internet_download', 'internet_upload']
X_test = df[features]
X_test_scaled = scaler.transform(X_test)

# === Tahmin ve Anomali Tespiti ===
df['cluster'] = kmeans.predict(X_test_scaled)
test_distances = cdist(X_test_scaled, kmeans.cluster_centers_, 'euclidean')
df['distance_to_center'] = test_distances.min(axis=1)
df['is_anomaly'] = df['distance_to_center'] > threshold

# === Küme Uzaklıklarına Göre Kümeleri Etiketle (Normal / Suspicious / Fraud)
cluster_distances = {}
for c in range(kmeans.n_clusters):
    cluster_points = X_test_scaled[df['cluster'] == c]
    dists = cdist(cluster_points, [kmeans.cluster_centers_[c]], 'euclidean')
    cluster_distances[c] = dists.mean()

sorted_clusters = sorted(cluster_distances.items(), key=lambda x: x[1])
cluster_labels = {}
cluster_labels[sorted_clusters[0][0]] = 'normal'
cluster_labels[sorted_clusters[1][0]] = 'suspicious'
cluster_labels[sorted_clusters[2][0]] = 'fraud'

df['label'] = df['cluster'].map(cluster_labels)

# Kullanıcı başına etiket dağılımı
print("\n📊 Kullanıcı Etiket Dağılımı:")
for label in ['normal', 'suspicious', 'fraud']:
    user_count = df[df['label'] == label]['user_id'].nunique()
    print(f"  - {label.capitalize():<10}: {user_count} kullanıcı")

# === Renk haritası
label_colors = {
    'normal': 'lightgray',
    'suspicious': 'orange',
    'fraud': 'red'
}
colors = df['label'].map(label_colors)

# === Görselleştirme
plt.figure(figsize=(15, 6))
plt.scatter(df['timestamp'], df['total_traffic_score'], c=colors, s=12)
plt.xlabel("Zaman")
plt.ylabel("Toplam Trafik Skoru")
plt.title("Küme Bazlı Anomali Etiketi: Normal / Şüpheli / Fraud")
plt.grid(True)
plt.tight_layout()
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label.capitalize(),
                              markerfacecolor=clr, markersize=10)
                   for label, clr in label_colors.items()]
plt.legend(handles=legend_elements)
plt.show()

# === Kullanıcı Bazlı Etiketleme (Genel Davranışa Göre)
def assign_user_label(user_df):
    counts = user_df['label'].value_counts()
    return counts.idxmax()  # en çok geçen etiketi döndür

user_labels = df.groupby('user_id').apply(assign_user_label).rename("user_label")
df = df.merge(user_labels, on='user_id')

# === Yeni Renk haritası (kullanıcı bazlı)
user_label_colors = {
    'normal': 'lightgray',
    'suspicious': 'orange',
    'fraud': 'red'
}
user_colors = df['user_label'].map(user_label_colors)

# === Görselleştirme (Kullanıcı Bazlı Etiketle)
plt.figure(figsize=(15, 6))
plt.scatter(df['timestamp'], df['total_traffic_score'], c=user_colors, s=12)
plt.xlabel("Zaman")
plt.ylabel("Toplam Trafik Skoru")
plt.title("Kullanıcı Bazlı Etiketleme: Normal / Şüpheli / Fraud")
plt.grid(True)
plt.tight_layout()
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label.capitalize(),
                              markerfacecolor=clr, markersize=10)
                   for label, clr in user_label_colors.items()]
plt.legend(handles=legend_elements)
plt.show()

# Kullanıcı etiketi özet bilgisi
print("\n📊 Kullanıcı Bazlı Genel Etiket Dağılımı:")
for label in ['normal', 'suspicious', 'fraud']:
    count = (user_labels == label).sum()
    print(f"  - {label.capitalize():<10}: {count} kullanıcı")