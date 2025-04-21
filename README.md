# 🧠 DBSCAN-Business-Segmentation

Bu proje, Northwind veritabanı üzerinde **DBSCAN algoritması** kullanılarak farklı iş problemlerine yönelik **kümeleme (clustering)** çözümleri sunar.

## 🎯 Amaç

Veri madenciliği ve segmentasyon tekniklerini kullanarak:

- **Anormal davranışları** tespit etmek  
- **Az katkı sağlayan grupları** belirlemek  
- **İş kararları için içgörü üretmek**

## 📊 Problemler ve Hedefler

### 📌 1. Ürün Segmentasyonu  
**Dosya:** `product_clustering/dbscan_product_clustering.py`

- **Problem:** Benzer sipariş geçmişine sahip ürünleri gruplandırın.
- **Özellikler:**
  - Ortalama satış fiyatı  
  - Satış sıklığı  
  - Sipariş başına ortalama miktar  
  - Kaç farklı müşteriye satıldı
- **Amaç:** Niş ürünleri ve az satılan ürünleri tespit etmek

---

### 📌 2. Tedarikçi Segmentasyonu  
**Dosya:** `supplier_clustering/dbscan_supplier_segment_analysis.py`

- **Problem:** Tedarikçileri, ürünlerinin satış performansına göre grupla.
- **Özellikler:**
  - Tedarik ettiği ürün sayısı  
  - Toplam satış miktarı  
  - Ortalama satış fiyatı  
  - Ortalama müşteri sayısı
- **Amaç:** Az katkı sağlayan ya da sıra dışı tedarikçileri tespit etmek

---

### 📌 3. Ülkelere Göre Satış Deseni Analizi  
**Dosya:** `country_sales_clustering/dbscan_country_sales_analysis.py`

- **Problem:** Farklı ülkelerin sipariş alışkanlıklarını gruplandırın.
- **Özellikler:**
  - Toplam sipariş sayısı  
  - Ortalama sipariş tutarı  
  - Sipariş başına ürün sayısı
- **Amaç:** Alışılmadık sipariş alışkanlığı gösteren ülkeleri belirlemek

---

## 🔌 Veritabanı Bağlantısı

Analiz dosyaları, Northwind veritabanına SQLAlchemy üzerinden bağlanarak çalışır. Aşağıdaki örnek bağlantı parametrelerini kendi sisteminize göre **güncellemeniz gerekir**:

```python
# Veritabanı bağlantısı
user = "postgres"           # PostgreSQL kullanıcı adınız
password = "sifre"          # Şifreniz
host = "localhost"          # Sunucu (genellikle localhost)
port = "5432"               # Port (PostgreSQL için varsayılan 5432)
database = "GYK1Northwind"  # Kullanacağınız veritabanı ismi

from sqlalchemy import create_engine

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

```

---

## ⚠️ Not

- PostgreSQL kurulu değilse bağlantı hatası alırsınız. Ayrıca, psycopg2 kütüphanesinin kurulu olduğundan emin olun:

  ```bash
  pip install psycopg2-binary
    ```



## 🛠️ Kullanılan Teknolojiler

- Python
- PostgreSQL (Northwind DB)
- SQLAlchemy
- Scikit-learn (DBSCAN, PCA)
- Kneed (Elbow metodu)
- Matplotlib, Pandas, NumPy

---

## 📁 Klasör Yapısı

```
DBSCAN-Business-Segmentation/
├── product_clustering/
│   └── dbscan_product_clustering.py
├── supplier_clustering/
│   └── dbscan_supplier_segment_analysis.py
├── country_sales_clustering/
│   └── dbscan_country_sales_analysis.py
└── README.md
```

---

## 📌 Notlar

- Her analiz kendi klasöründe bağımsız olarak çalıştırılabilir.
- EPS değeri otomatik olarak Elbow yöntemiyle belirlenmektedir.
- `-1` olan kümeler, **aykırı grupları (outliers)** temsil eder.

---
