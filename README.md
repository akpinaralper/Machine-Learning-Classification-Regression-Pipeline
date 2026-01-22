# 📊 Makine Öğrenmesi Sınıflandırma & Regresyon Hattı  
**Dry Bean & Algerian Forest Fires Veri Seti Analizi**

Bu proje, iki farklı gerçek dünya veri seti üzerinde sınıflandırma ve regresyon problemlerini çözmek için modern makine öğrenmesi algoritmalarını kullanan uçtan uca bir analiz hattıdır.  

Amaç:  
Veri indirme → temizleme → modelleme → çapraz doğrulama → performans karşılaştırması → görselleştirme  
adımlarını tamamen otomatik ve sunuma hazır şekilde üretmek.

---

## 🚀 Proje Özeti

Proje iki ana bölümden oluşmaktadır:

### 🔹 Bölüm 1 — Sınıflandırma (Dry Bean Veri Seti)
- Amaç: Fasulye türlerini sınıflandırmak  
- Veri Kaynağı: UCI Machine Learning Repository – Dry Bean Dataset  
- Toplam Örnek Sayısı: 13.611  
- Özellik Sayısı: 16    
- Sınıflar: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY  
- Kullanılan Modeller:
  - SVM (RBF Kernel)  
  - XGBoost Classifier  
- Değerlendirme Metrikleri:
  - Accuracy (Doğruluk)  
  - F1-Score (Weighted)  
---

### 🔹 Bölüm 2 — Regresyon (Algerian Forest Fires Veri Seti)
- Amaç: Fire Weather Index (FWI) tahmini  
- Veri Kaynağı: UCI Machine Learning Repository – Algerian Forest Fires  
- Temizlenmiş Örnek Sayısı: 243   
- Kullanılan Modeller:
  - SVR (RBF Kernel)  
  - XGBoost Regressor  
- Değerlendirme Metrikleri:
  - MAE (Mean Absolute Error)  
  - SMAPE (Symmetric Mean Absolute Percentage Error)  
---



##  Projeyi Çalıştırma Sonrası

Projeyi çalıştırdığınızda:

- Veri setleri otomatik olarak indirilir  
- Veri temizleme ve ölçekleme yapılır  
- Modeller 3-Fold Cross Validation ile eğitilir  
- Performans metrikleri hesaplanır  
- Grafikler otomatik olarak gösterilir  
- En sonda genel karşılaştırma tablosu terminale yazdırılır  


## Örnek Çıktılar ve Görselleştirmeler

### Dry Bean – Sınıf Dağılımı
<img width="1000" height="500" alt="drybean_histogram" src="https://github.com/user-attachments/assets/d5e156fd-14e9-428a-b065-fabf6106b10e" />

### SVM – Hata Matrisi

<img width="800" height="600" alt="svm_confusion_matrix" src="https://github.com/user-attachments/assets/6dec3fb4-0935-4f3d-a296-7a68a1e73879" />

### XGBoost – Hata Matrisi
<img width="800" height="600" alt="xgboost_confusion_matrix" src="https://github.com/user-attachments/assets/d3cb26c4-4dcd-4c45-b9cf-9422cb869d83" />

### FWI Hedef Değişken Dağılımı
<img width="1000" height="500" alt="fwi_histogram" src="https://github.com/user-attachments/assets/9f588396-52b5-4b00-84ec-6feaa5e2b011" />

### SVR – Gerçek vs Tahmin
<img width="700" height="700" alt="svr_prediction" src="https://github.com/user-attachments/assets/7ad12ebc-c851-428f-a5bf-69c449455d09" />

### XGBoost – Gerçek vs Tahmin
<img width="700" height="700" alt="xgboost_prediction" src="https://github.com/user-attachments/assets/6d959781-f5b6-4d18-8160-e1574428b5bb" />


## 📈 Model Performans Sonuçları

### 🔹 Sınıflandırma — Dry Bean Veri Seti

| Model   | Accuracy | F1-Score |
|---------|----------|----------|
| SVM     | %92.84   | %92.85   |
| XGBoost | %92.40   | %92.40   |

---

### 🔹 Regresyon — Algerian Forest Fires Veri Seti

| Model   | MAE     | SMAPE   |
|---------|---------|---------|
| SVR     | 1.5800  | %48.31  |
| XGBoost | 0.7127  | %23.82  |

---

## 🏆 Genel Karşılaştırma Tablosu

| Veri Seti                   | Model   | 1. Metrik        | 2. Metrik         |
|-----------------------------|---------|------------------|-------------------|
| Dry Bean (Sınıflandırma)    | SVM     | %92.84 (ACC)    | %92.85 (F1)      |
| Dry Bean (Sınıflandırma)    | XGBoost | %92.40 (ACC)    | %92.40 (F1)      |
| Algerian Forest (Regresyon) | SVR     | 1.5800 (MAE)    | %48.31 (SMAPE)   |
| Algerian Forest (Regresyon) | XGBoost | 0.7127 (MAE)    | %23.82 (SMAPE)   |







## Temel İstatistik Özeti
| Veri Seti       | Örnek Sayısı | Özellik Sayısı | Hedef |
| --------------- | ------------ | -------------- | ----- |
| Dry Bean        | 13.611       | 16             | Class |
| Algerian Forest | 243          | —              | FWI   |




##  Temel Çıkarımlar

- Dry Bean veri seti dengesizdir (en büyük sınıf, en küçüğün 6.8 katıdır)  
- Buna rağmen SVM ve XGBoost %92 üzeri doğruluk elde etmiştir  
- XGBoost, sınıflandırmada SVM’den az farkla geride kalmıştır  
- Regresyonda ise SVR’ye kıyasla çok daha düşük hata üretmiştir  
- FWI dağılımı sağa çarpıktır, bu yüzden MAE metriği daha güvenilirdir  
- XGBoost regresyonda yaklaşık %55 daha düşük MAE elde etmiştir  


## 📌 Notlar

- Dry Bean veri seti dengesizdir (en büyük sınıf, en küçüğün 6.8 katıdır)  
- Büyük veri setlerinde görseller için rastgele 1000 örnek kullanılır  
- XGBoost bu projede regresyon için en başarılı modeldir  





##  Lisans

Bu proje Örüntü Tanıma dersi kapsamında hazırlanmıştır, eğitim ve araştırma amaçlıdır.  
Serbestçe geliştirilebilir ve yeniden kullanılabilir.
