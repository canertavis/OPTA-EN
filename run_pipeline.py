"""
=============================================================================
ANA PIPELINE — 11-17 Mayıs 2026 Talep Tahmini ve Optimizasyon
=============================================================================
Kullanım:
  python run_pipeline.py               → tüm adımları çalıştır
  python run_pipeline.py --skip-tuning → hyperparameter tuning'i atla
  python run_pipeline.py --dashboard   → bitince dashboard'u aç

Adım sırası:
  1. data_loader.py            → Excel dosyalarını hazırla
  2. demand_forecast.py        → baseline XGBoost modelleri + 11-17 Mayıs tahmini
  3. hyperparameter_tuning.py  → modelleri Optuna ile iyileştir + tahmini güncelle
  4. linehaul_optimization_.py → Gurobi MILP spot araç optimizasyonu
=============================================================================
"""

import subprocess, sys, time, os
from datetime import datetime

def adim_calistir(dosya, aciklama):
    print(f"\n{'━' * 65}")
    print(f"  ▶  {aciklama}")
    print(f"     {dosya}")
    print(f"{'━' * 65}")
    baslangic = time.time()
    sonuc = subprocess.run([sys.executable, dosya])
    sure = time.time() - baslangic
    if sonuc.returncode != 0:
        print(f"\n  ✗  HATA: {dosya} başarısız oldu (kod: {sonuc.returncode})")
        print(f"     Pipeline durduruluyor.")
        sys.exit(1)
    print(f"\n  ✓  Tamamlandı — {sure:.1f} saniye")
    return sure

def dosya_kontrol(yol, aciklama):
    if os.path.exists(yol):
        boyut = os.path.getsize(yol)
        print(f"  ✓  {aciklama} ({boyut:,} byte)")
    else:
        print(f"  ✗  EKSİK: {aciklama} → {yol}")

skip_tuning  = "--skip-tuning"  in sys.argv
ac_dashboard = "--dashboard"    in sys.argv
toplam       = 0

print(f"\n{'═' * 65}")
print(f"  ANAHAT LOJİSTİK OPTİMİZASYONU — Pipeline Başlıyor")
print(f"  Tarih  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Tuning : {'atlanıyor (--skip-tuning)' if skip_tuning else 'çalışacak'}")
print(f"{'═' * 65}")

toplam += adim_calistir("data_loader.py",
    "ADIM 1 / 5 — Excel Verilerini Yükle ve Hazırla")

toplam += adim_calistir("demand_forecast.py",
    "ADIM 2 / 5 — Baseline XGBoost Modelleri + 11-17 Mayıs Tahmini")

if skip_tuning:
    print(f"\n  ⏭  Adım 3 atlandı (--skip-tuning)")
else:
    toplam += adim_calistir("hyperparameter_tuning.py",
        "ADIM 3 / 5 — Optuna Hiperparametre Optimizasyonu + Tahmin Güncelle")

toplam += adim_calistir("linehaul_optimization.py",
    "ADIM 4 / 5 — Gurobi MILP Spot Araç Optimizasyonu")

toplam += adim_calistir("export_outputs.py",
    "ADIM 5 / 5 — Teknofest Çıktıları Üret (Excel + README)")

print(f"\n{'═' * 65}")
print("  ÇIKTI DOĞRULAMA")
print(f"{'═' * 65}")
dosya_kontrol("data/talep.csv",                         "Tarihsel talep verisi")
dosya_kontrol("data/mesafeler.csv",                     "Rota mesafeleri")
dosya_kontrol("predictions/haftalik_tahminler.csv",     "11-17 Mayıs tahminleri")
dosya_kontrol("predictions/model_metrikleri.csv",       "Baseline model metrikleri")
dosya_kontrol("predictions/tuning_sonuclari.csv",       "Tuning karşılaştırma tablosu")
dosya_kontrol("predictions/optimizasyon_sonuclari.csv", "Spot araç optimizasyon planı")
dosya_kontrol("predictions/ozet_metrikler.csv",         "Özet maliyet metrikleri")
dosya_kontrol("models/best_params.json",                "En iyi hiperparametreler")
dosya_kontrol("outputs/tahmin_ciktisi.xlsx",            "📊 Teknofest: Tahmin çıktısı (Excel)")
dosya_kontrol("outputs/arac_planlama.xlsx",             "🚛 Teknofest: Araç planlama (Excel)")
dosya_kontrol("outputs/README.md",                      "📋 Teknofest: Maliyet özeti (README)")

print(f"\n{'═' * 65}")
print(f"  PİPELİNE TAMAMLANDI — {toplam:.1f} saniye ({toplam/60:.1f} dakika)")
print(f"{'═' * 65}")
print(f"\n  Dashboard için: streamlit run dashboard.py")

if ac_dashboard:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
