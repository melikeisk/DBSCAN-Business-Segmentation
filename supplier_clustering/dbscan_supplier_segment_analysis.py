import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Bu satır eklendi
from sqlalchemy import create_engine
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Veritabanı bağlantısı
user = "postgres"
password = "sifre"
host = "localhost"
port = "5432"
database = "GYK1Northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# Veritabanından Tedarikçi ve Ürün bilgisi çekme
query = """
SELECT
    s.supplier_id,
    COUNT(p.product_id) AS product_count,
    SUM(od.quantity) AS total_sales,
    AVG(od.unit_price) AS avg_price,
    COUNT(DISTINCT o.customer_id) AS avg_customer_count
FROM
    suppliers s
JOIN
    products p ON s.supplier_id = p.supplier_id
JOIN
    order_details od ON p.product_id = od.product_id
JOIN
    orders o ON od.order_id = o.order_id
GROUP BY
    s.supplier_id
"""
df = pd.read_sql_query(query, engine)

# Özelliklerin ölçeklendirilmesi (Standardization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['product_count', 'total_sales', 'avg_price', 'avg_customer_count']])

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

# 2. DBSCAN ile tedarikçileri grupla
model = DBSCAN(eps=eps, min_samples=3, metric='euclidean')  # Euclidean mesafesi ile
labels = model.fit_predict(df_scaled)

# 3. Sonuçları birleştir
df['cluster'] = labels
print(df['cluster'].value_counts())

# 4. Aykırı tedarikçiler (-1 olanlar)
outliers = df[df['cluster'] == -1]
print(f"Aykırı tedarikçi sayısı: {len(outliers)}")
print(outliers[['supplier_id', 'product_count', 'total_sales', 'avg_price', 'avg_customer_count']])

# 5. Segmentler arasındaki farkları inceleyelim (grupların özellikleri)
cluster_summary = df.groupby('cluster').agg({
    'product_count': 'mean',
    'total_sales': 'mean',
    'avg_price': 'mean',
    'avg_customer_count': 'mean'
})
print(cluster_summary)

# 6. PCA ile Boyut İndirgeme ve Kümeleme Sonuçlarını Görselleştirme
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='plasma', s=60)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA ile Kümeleme Sonuçları')
plt.colorbar(label='Küme No')
plt.show()

# 7. Aykırı Tedarikçilerin Görselleştirilmesi
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='plasma', s=60, label='Normal Tedarikçiler')
plt.scatter(reduced_data[outliers.index, 0], reduced_data[outliers.index, 1], color='red', label='Aykırı Tedarikçiler', s=100)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Aykırı Tedarikçilerin Görselleştirilmesi')
plt.legend()
plt.show()
