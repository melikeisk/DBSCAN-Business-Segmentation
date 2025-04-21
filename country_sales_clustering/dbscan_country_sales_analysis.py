import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler

# Veritabanı bağlantısı
user = "postgres"
password = "sifre"
host = "localhost"
port = "5432"
database = "GYK1Northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# Veritabanından Gerekli Bilgileri Çekme
query = """
SELECT
    c.country,
    COUNT(o.order_id) AS total_orders,
    AVG(od.unit_price * od.quantity) AS avg_order_value,  -- Sipariş başına ortalama tutar
    AVG(od.quantity) AS avg_products_per_order
FROM
    customers c
JOIN
    orders o ON c.customer_id = o.customer_id
JOIN
    order_details od ON o.order_id = od.order_id
GROUP BY
    c.country
"""
df = pd.read_sql_query(query, engine)

# Özelliklerin ölçeklendirilmesi (Standardization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['total_orders', 'avg_order_value', 'avg_products_per_order']])

# 1. Optimal eps değeri bul (Elbow yöntemi ile)
def find_optimal_eps(X, min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = neighbors.kneighbors(X)
    distances = np.sort(distances[:, min_samples - 1])

    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]

    plt.figure(figsize=(10, 6))  # Şimdi plt doğru şekilde tanımlandı
    plt.plot(distances)
    plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.xlabel('Veri Nokaları (Sıralı)')
    plt.ylabel(f'{min_samples}. En Yakın Komşu Mesafesi')
    plt.title('Elbow Yöntemi ile Optimal eps')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_eps

eps = find_optimal_eps(df_scaled, min_samples=3)

# 2. DBSCAN ile ülkeleri grupla
model = DBSCAN(eps=eps, min_samples=3, metric='euclidean')  # Euclidean mesafesi ile
labels = model.fit_predict(df_scaled)

# 3. Sonuçları birleştir
df['cluster'] = labels
print(df['cluster'].value_counts())

# 4. Aykırı ülkeler (-1 olanlar)
outliers = df[df['cluster'] == -1]
print(f"Aykırı ülke sayısı: {len(outliers)}")
print(outliers[['country', 'total_orders', 'avg_order_value', 'avg_products_per_order']])

# 5. Segmentler arasındaki farkları inceleyelim (grupların özellikleri)
cluster_summary = df.groupby('cluster').agg({
    'total_orders': 'mean',
    'avg_order_value': 'mean',
    'avg_products_per_order': 'mean'
})
print(cluster_summary)
