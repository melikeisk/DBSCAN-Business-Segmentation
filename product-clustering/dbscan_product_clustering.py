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
password = "1881"
host = "localhost"
port = "5432"
database = "GYK1Northwind"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

# Veriyi çek
query = """
SELECT
    p.product_id,
    AVG(od.unit_price) AS avg_price,
    COUNT(od.order_id) AS order_count,
    AVG(od.quantity) AS avg_quantity,
    COUNT(DISTINCT o.customer_id) AS unique_customers
FROM
    products p
JOIN
    order_details od ON p.product_id = od.product_id
JOIN
    orders o ON od.order_id = o.order_id
GROUP BY
    p.product_id
"""
df = pd.read_sql_query(query, engine)

# Özellik vektörlerini standardize et
scaler = StandardScaler()
X = scaler.fit_transform(df[['avg_price', 'order_count', 'avg_quantity', 'unique_customers']])

# Optimal eps değerini bulmak için Elbow yöntemi
def find_optimal_eps(X, min_samples=3):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = neighbors.kneighbors(X)
    distances = np.sort(distances[:, min_samples - 1])
    
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]

    plt.figure(figsize=(8, 5))
    plt.plot(distances, label='Distance to Nearest Neighbors')
    plt.axvline(x=kneedle.elbow, color='red', linestyle='--', label=f'Optimal eps: {optimal_eps:.2f}')
    plt.title("Elbow Yöntemi ile eps Belirleme")
    plt.xlabel("Veri Nokası Sırası")
    plt.ylabel("Mesafe")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return optimal_eps

# eps değerini bul
eps = find_optimal_eps(X, min_samples=3)

# DBSCAN modeli
model = DBSCAN(eps=eps, min_samples=3)
labels = model.fit_predict(X)

# Sonuçları dataframe'e ekle
df['cluster'] = labels

# Küme dağılımı
print(df['cluster'].value_counts())

# Aykırı ürünler
outliers = df[df['cluster'] == -1]
print(f"\nAykırı ürün sayısı: {len(outliers)}")
print(outliers[['product_id', 'avg_price', 'order_count', 'avg_quantity', 'unique_customers']])

# Küme özetleri
cluster_summary = df.groupby('cluster').agg({
    'avg_price': 'mean',
    'order_count': 'mean',
    'avg_quantity': 'mean',
    'unique_customers': 'mean'
})
print("\nKüme Özetleri:")
print(cluster_summary)
