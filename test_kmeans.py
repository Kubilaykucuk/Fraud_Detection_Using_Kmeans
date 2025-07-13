import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

# === Ayarlar ===
data_folder = 'datas'
test_file = os.path.join(data_folder, 'sms-call-internet-mi-2013-11-10.txt')
model_path = 'kmeans_model.pkl'
scaler_path = 'scaler.pkl'
threshold_path = 'threshold.pkl'

# === Model ve Scaler yükle ===
print("📦 Model ve Scaler yükleniyor...")
kmeans = joblib.load(model_path)
scaler = joblib.load(scaler_path)
threshold = joblib.load(threshold_path)

# === Test verisini oku ve hazırla ===
print(f"📂 Test verisi okunuyor: {os.path.basename(test_file)}")
df = pd.read_csv(test_file, sep="\t", header=None)
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

# === Özellik seçimi ve ölçekleme
features = ['duration', 'sms_volume', 'call_volume', 'internet_download', 'internet_upload']
X_test = df[features]
X_scaled = scaler.transform(X_test)

# === K-means cluster atama
df['cluster'] = kmeans.predict(X_scaled)

# === Cluster davranış skorları
cluster_distances = {}
for c in range(kmeans.n_clusters):
    cluster_points = X_scaled[df['cluster'] == c]
    dists = cdist(cluster_points, [kmeans.cluster_centers_[c]], 'euclidean')
    cluster_distances[c] = dists.mean()

# === Cluster etiketleme
sorted_clusters = sorted(cluster_distances.items(), key=lambda x: x[1])
cluster_labels = {
    sorted_clusters[0][0]: 'normal',
    sorted_clusters[1][0]: 'suspicious',
    sorted_clusters[2][0]: 'fraud'
}
df['label'] = df['cluster'].map(cluster_labels)

# === 📌 Distance to center kullanılmadan is_anomaly belirle
df['is_anomaly'] = df['label'] == 'fraud'

# === 🎨 Distance to center olmadan örnek çizimi
print("🎨 Distance to center olmadan örnek çiziliyor...")
plt.figure(figsize=(14, 6))
plt.scatter(df['timestamp'], df['total_traffic_score'], c=df['is_anomaly'], cmap='coolwarm', s=10)
plt.title("🔎 Anomali Tespiti (Sadece Cluster Etiketine Göre)")
plt.xlabel("Zaman")
plt.ylabel("Toplam Trafik Skoru")
plt.grid(True)
plt.tight_layout()

# === Kullanıcı bazlı kişilik etiketi
def assign_user_label(group):
    return group['label'].value_counts().idxmax()

user_labels = df.groupby('user_id').apply(assign_user_label).rename("user_label")
df = df.merge(user_labels, on='user_id')

# === 🎨 Kullanıcı bazlı kişilik etiketi çizimi
label_colors = {'normal': 'lightgray', 'suspicious': 'orange', 'fraud': 'red'}
df['color'] = df['user_label'].map(label_colors)

print("👥 Kullanıcı bazlı kişilik etiketi çiziliyor (Plotly)...")

plt.figure(figsize=(14, 6))
plt.scatter(df['timestamp'], df['total_traffic_score'], c=df['color'], s=10)
plt.title("👥 Kullanıcı Bazlı Kişilik Etiketi")
plt.xlabel("Zaman")
plt.ylabel("Toplam Trafik Skoru")
plt.grid(True)
plt.tight_layout()

# === 🧮 Etiket dağılımı çıktısı
label_counts = df['user_label'].value_counts()
print("📌 Etiket Dağılımı (Kullanıcı Bazlı):")
user_counts = df[['user_id', 'user_label']].drop_duplicates()
for label in ['normal', 'suspicious', 'fraud']:
    count = (user_counts['user_label'] == label).sum()
    print(f"  - {label.capitalize():<10}: {count} kullanıcı")

# === 📈 Interaktif çizim: Fraud kullanıcıların trafik paternleri
print("📊 Etkileşimli fraud kullanıcı çizimi hazırlanıyor...")
fraud_df = df[df['user_label'] == 'fraud'].copy()
fraud_df['timestamp'] = fraud_df['timestamp'].astype(str)

fig_fraud = px.line(fraud_df,
              x='timestamp',
              y='total_traffic_score',
              color='user_id',
              title="📈 Fraud Kullanıcıların Trafik Paternleri (Seçilebilir)",
              labels={'total_traffic_score': 'Toplam Trafik Skoru', 'timestamp': 'Zaman'},
              template='plotly_white')

fig_fraud.update_layout(
    legend_title_text='Kullanıcı ID',
    hovermode='x unified'
)

fig_fraud.show()

# === 📈 Etkileşimli çizim: Suspicious kullanıcıların trafik paternleri
print("📊 Etkileşimli suspicious kullanıcı çizimi hazırlanıyor...")
suspicious_df = df[df['user_label'] == 'suspicious'].copy()
suspicious_df['timestamp'] = suspicious_df['timestamp'].astype(str)

fig_susp = px.line(
    suspicious_df,
    x='timestamp',
    y='total_traffic_score',
    color='user_id',
    title="📈 Suspicious Kullanıcıların Trafik Paternleri (Seçilebilir)",
    labels={'total_traffic_score': 'Toplam Trafik Skoru', 'timestamp': 'Zaman'},
    template='plotly_white'
)

fig_susp.update_layout(
    legend_title_text='Kullanıcı ID',
    hovermode='x unified'
)

fig_susp.show()