# Anahat Lojistik Optimizasyon Sistemi
## Teknofest 2026 — Hepsiburada İş Birliği

---

## Sistem Mimarisi

```text
Tarihsel Veri (CSV)
        ↓
Feature Engineering (Zaman Serisi + Zengin Bağlam)
        ↓
XGBoost vs Mevsimsel Naif (AutoML)
        ↓
Optuna Hiperparametre Tuning (TPE)
        ↓
Gurobi MILP Optimizasyonu (Hesaplanmış Risk & Volume-Proportional Dynamic Buffer)
        ↓
Streamlit Dashboard 
```

---

## 11-17 Mayıs 2026 — Sonuçlar

### Talep Tahmini

|    Metrik    |    Değer    |
|--------------|-------------|
| Tahmin yapılan rota | 89 |
| Ortalama MAPE | %25.9 |
| Ortalama R² | 0.681 |
| XGBoost (Set A) seçilen | 26 rota |
| XGBoost (Set B) seçilen | 38 rota |
| Mevsimsel Naif seçilen | 89 rota |
| Haftalık toplam tahmin | 7,310,539 desi |

### Maliyet Bilgisi

|   Metrik     |    Değer    |
|--------------|-------------|
| **Toplam Lojistik Maliyeti** | **₺10,722,750** |
| Baseline Maliyet (tüm spot, kamyonet) | ₺17,061,453 |
| **Sağlanan Tasarruf** | **₺6,338,703** |
| **Tasarruf Oranı** | **%37.2** |
| Kiralık Filo Karşıladığı | 1,904,000 desi |
| Spot Araçla Karşılanan | 7,029,272 desi |
| Toplam Spot Araç Kullanımı | 654 araç/gün |

---

## Çıktı Dosyaları

| Dosya | İçerik |
|-------|--------|
| `outputs/tahmin_ciktisi.xlsx` | Günlük rota bazlı talep tahminleri, haftalık pivot, model metrikleri |
| `outputs/arac_planlama.xlsx` | Gün-rota bazlı araç planı, özet, araç tipi dağılımı |
| `outputs/README.md` | Bu dosya — toplam maliyet ve özet sonuçlar |

---

## Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Veri işleme | pandas, numpy |
| ML modeli | XGBoost, scikit-learn |
| Hiperparametre | Optuna (TPE) |
| Optimizasyon | Gurobi (gurobipy) |
| Dashboard | Streamlit, Plotly |
| Excel çıktı | openpyxl |

---

## Çalıştırma

```bash
# Tüm pipeline
python run_pipeline.py

# Sadece çıktı üret (pipeline çalıştırıldıktan sonra)
python export_outputs.py
```

---

*Oluşturma tarihi: 21.06.2026 17:27*
