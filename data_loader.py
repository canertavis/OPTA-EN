"""
=============================================================================
VERİ YÜKLEYİCİ — Excel Dosyalarını Pipeline'a Hazırla
=============================================================================
Amaç  : 4 Excel dosyasını okuyup data/ klasörüne CSV olarak kaydeder.
Çıktı : data/talep.csv          → tarihsel talep verisi
        data/arac_maliyet.csv   → araç kapasite ve maliyet tablosu
        data/kiralik_arac.csv   → mevcut kiralık filo
        data/koordinatlar.csv   → transfer merkezi koordinatları
        data/mesafeler.csv      → rota bazında hesaplanan km mesafeleri

Çalıştırma sırası:
  1. python data_loader.py          ← bu dosya
  2. python demand_forecast.py
  3. python linehaul_optimization_.py
=============================================================================
"""

import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
os.makedirs("predictions", exist_ok=True) 

# =============================================================================
# BÖLÜM 1 — EXCEL DOSYALARINI OKU
# =============================================================================

TALEP_DOSYA    = "veriseti/Desi_talep.xlsx"
ARAC_DOSYA     = "veriseti/Arac_Kapasite_Maliyet.xlsx"
KIRALIK_DOSYA  = "veriseti/Kiralık_Araclar.xlsx"
KOORD_DOSYA    = "veriseti/Koordinatlar.xlsx"

print("  Veri dosyaları yükleniyor...")

talep_df   = pd.read_excel(TALEP_DOSYA)
arac_df    = pd.read_excel(ARAC_DOSYA)
kiralik_df = pd.read_excel(KIRALIK_DOSYA)
koord_df   = pd.read_excel(KOORD_DOSYA)

# =============================================================================
# BÖLÜM 2 — TALEP VERİSİ
# =============================================================================

talep_df.columns = ["cikis", "varis", "tarih", "talep_desi"]
talep_df["tarih"] = pd.to_datetime(talep_df["tarih"])
talep_df["rota"]  = talep_df["cikis"] + "→" + talep_df["varis"]

# Takvim özellikleri ekle
talep_df["haftanin_gunu"] = talep_df["tarih"].dt.dayofweek   # 0=Pzt, 6=Paz
talep_df["ay"]            = talep_df["tarih"].dt.month
talep_df["yil"]           = talep_df["tarih"].dt.year
talep_df["gun_no"]        = talep_df["tarih"].dt.dayofyear
talep_df["hafta_no"]      = talep_df["tarih"].dt.isocalendar().week.astype(int)
talep_df["hafta_sonu"]    = (talep_df["haftanin_gunu"] >= 5).astype(int)
talep_df["ay_sonu_mu"]    = (talep_df["tarih"].dt.day >= 28).astype(int)
talep_df["tatil_mi"]      = 0   # Gerçek tatil verisi olmadığı için sıfır

talep_df.to_csv("data/talep.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ Talep verisi: {len(talep_df):,} satır, {talep_df['rota'].nunique()} rota")
print(f"    Tarih aralığı: {talep_df['tarih'].min().date()} → {talep_df['tarih'].max().date()}")

# =============================================================================
# BÖLÜM 3 — ARAÇ MALİYET TABLOSU
# =============================================================================

arac_df.columns = [
    "arac_tipi", "kapasite_desi",
    "kiralik_gunluk_tl", "kiralik_km_tl",
    "spot_gunluk_tl", "spot_km_tl"
]
arac_df.to_csv("data/arac_maliyet.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ Araç maliyet tablosu: {len(arac_df)} araç tipi")

# =============================================================================
# BÖLÜM 4 — KİRALIK FİLO
# =============================================================================

kiralik_df.columns = ["cikis", "varis", "arac_sayisi", "arac_tipi"]
kiralik_df["rota"] = kiralik_df["cikis"] + "→" + kiralik_df["varis"]

# Kiralık filo kapasitesini hesapla (araç tipi → kapasite)
kapasite_map = dict(zip(arac_df["arac_tipi"], arac_df["kapasite_desi"]))
kiralik_df["kapasite_desi"] = (
    kiralik_df["arac_tipi"].map(kapasite_map) * kiralik_df["arac_sayisi"]
)
kiralik_df.to_csv("data/kiralik_arac.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ Kiralık filo: {len(kiralik_df)} rota, toplam {kiralik_df['arac_sayisi'].sum()} araç")

# =============================================================================
# BÖLÜM 5 — KOORDİNATLAR
# =============================================================================

koord_df.columns = ["sehir", "enlem", "boylam"]
koord_df.to_csv("data/koordinatlar.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ Koordinatlar: {len(koord_df)} transfer merkezi")

# =============================================================================
# BÖLÜM 6 — ROTA MESAFE HESABI (Haversine)
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """İki koordinat arası mesafeyi km olarak hesapla."""
    R = 6371  # Dünya yarıçapı (km)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi       = np.radians(lat2 - lat1)
    dlambda    = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Koordinat sözlüğü
koord = dict(zip(koord_df["sehir"], zip(koord_df["enlem"], koord_df["boylam"])))

# Talep verisindeki tüm rotalar için mesafe hesapla
rotalar = talep_df[["cikis","varis","rota"]].drop_duplicates()
mesafeler = []

for _, row in rotalar.iterrows():
    cikis, varis = row["cikis"], row["varis"]
    if cikis in koord and varis in koord:
        lat1, lon1 = koord[cikis]
        lat2, lon2 = koord[varis]
        km = round(haversine(lat1, lon1, lat2, lon2), 1)
    else:
        km = None   # Koordinat bulunamazsa boş bırak
        print(f"    UYARI: {cikis} veya {varis} koordinatı bulunamadı")

    mesafeler.append({"rota": row["rota"], "cikis": cikis, "varis": varis, "mesafe_km": km})

mesafe_df = pd.DataFrame(mesafeler)
mesafe_df.to_csv("data/mesafeler.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ Mesafe tablosu: {len(mesafe_df)} rota hesaplandı")

# =============================================================================
# ÖZET
# =============================================================================

print("\n" + "=" * 55)
print("  VERİ YÜKLEME TAMAMLANDI")
print("=" * 55)
print(f"  data/talep.csv          → {len(talep_df):,} satır")
print(f"  data/arac_maliyet.csv   → {len(arac_df)} araç tipi")
print(f"  data/kiralik_arac.csv   → {len(kiralik_df)} rota")
print(f"  data/koordinatlar.csv   → {len(koord_df)} şehir")
print(f"  data/mesafeler.csv      → {len(mesafe_df)} rota mesafesi")
print(f"\n  Sonraki adım: python demand_forecast.py")
