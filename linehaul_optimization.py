"""
=============================================================================
ANAHAT LOJİSTİK OPTİMİZASYONU — Gurobi MILP (Risk-Aware & Dynamic Buffer)
=============================================================================
Amaç  : 11-17 Mayıs 2026 haftası için her rota ve her gün,
        kiralık filo kapasitesi aşılan talebi karşılamak üzere
        en ucuz spot araç kombinasyonunu bul.

YENİ STRATEJİLER:
  ✓ Hacim Orantılı Dinamik Tampon: Tampon miktarı, o günkü talebin %30'unu geçemez.
  ✓ Yumuşak Kısıt (Soft Constraint): Ufak taşmalar için araç israfı yerine risk alınır.
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import gurobipy as gp
from gurobipy import GRB

# =============================================================================
# BÖLÜM 1 — VERİLERİ YÜKLEYİN
# =============================================================================

tahmin_df  = pd.read_csv("predictions/haftalik_tahminler.csv", parse_dates=["tarih"])
arac_df    = pd.read_csv("data/arac_maliyet.csv")
kiralik_df = pd.read_csv("data/kiralik_arac.csv")
mesafe_df  = pd.read_csv("data/mesafeler.csv")

MAE_MAP  = {}
if os.path.exists("predictions/model_metrikleri.csv"):
    metrik_df = pd.read_csv("predictions/model_metrikleri.csv")
    MAE_MAP   = dict(zip(metrik_df["rota"], metrik_df["MAE"]))

TAMPON_CARPANI = {
    "yüksek" : 1.0,  
    "sinirda" : 0.5,   
    "guvenli" : 0.0,   
}

def risk_tamponu(rota, talep, kiralik_kap):
    """
    Rota için Hacim Orantılı Dinamik Güvenlik Tamponu hesapla.
    """
    mae     = MAE_MAP.get(rota, 0)
    boslik  = kiralik_kap - talep   

    # Eğer o gün talep sıfırsa veya model hatasızsa tampona gerek yok
    if mae == 0 or talep <= 0:
        return 0

    if boslik < mae:
        carpan = TAMPON_CARPANI["yüksek"]
    elif boslik < mae * 2:
        carpan = TAMPON_CARPANI["sinirda"]
    else:
        carpan = TAMPON_CARPANI["guvenli"]

    ham_tampon = mae * carpan
    
    # 🚀 MÜHENDİSLİK DOKUNUŞU: Hacim Orantılı Sınır (Dynamic Cap)
    # Tampon, hiçbir şartta o günkü talebin %30'unu geçemez!
    dinamik_tampon = min(ham_tampon, talep * 0.30)

    return round(dinamik_tampon)

ARAC_TIPLERI = list(arac_df["arac_tipi"])
KAPASITE     = dict(zip(arac_df["arac_tipi"], arac_df["kapasite_desi"]))
SPOT_GUNLUK  = dict(zip(arac_df["arac_tipi"], arac_df["spot_gunluk_tl"]))
SPOT_KM      = dict(zip(arac_df["arac_tipi"], arac_df["spot_km_tl"]))

KIRALIK_KAPASITE = dict(zip(kiralik_df["rota"], kiralik_df["kapasite_desi"]))
MESAFE = dict(zip(mesafe_df["rota"], mesafe_df["mesafe_km"]))

ROTALAR = sorted(tahmin_df["rota"].unique())
TARIHLER = sorted(tahmin_df["tarih"].unique())

TALEP = {}
for _, row in tahmin_df.iterrows():
    TALEP[(row["rota"], row["tarih"])] = row["tahmin_desi"]

print("=" * 65)
print("  ANAHAT OPTİMİZASYON BAŞLIYOR (Dynamic Buffer & Risk-Aware)")
print("=" * 65)

# =============================================================================
# BÖLÜM 2 — SPOT MALİYET
# =============================================================================

def spot_maliyet(rota, arac_tipi):
    km = MESAFE.get(rota, 0)
    return SPOT_GUNLUK[arac_tipi] + SPOT_KM[arac_tipi] * km

# =============================================================================
# BÖLÜM 3, 4, 5 — GUROBİ MODELİ VE YUMUŞAK KISITLAR (SOFT CONSTRAINTS)
# =============================================================================

print("\n  Gurobi modeli kuruluyor...")
model = gp.Model("anahat_lojistik_optimizasyon")
model.setParam("OutputFlag", 0)
model.setParam("TimeLimit", 120)

x = {}
alinan_risk = {} # YENİ: Hesaplanmış Risk Değişkeni

for r in ROTALAR:
    for t in TARIHLER:
        alinan_risk[r, t] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=0,
            name=f"risk_{r.replace('→','_')}_{str(t)[:10]}"
        )
        for v in ARAC_TIPLERI:
            x[r, t, v] = model.addVar(
                vtype=GRB.INTEGER, lb=0,
                name=f"x_{r.replace('→','_')}_{str(t)[:10]}_{v}"
            )

def risk_ceza(rota):
    # En küçük aracın birim desi maliyetinden %10 ucuz ceza
    maliyet = spot_maliyet(rota, "Kamyonet")
    return (maliyet / KAPASITE["Kamyonet"]) * 0.9

model.setObjective(
    gp.quicksum(x[r, t, v] * spot_maliyet(r, v) for r in ROTALAR for t in TARIHLER for v in ARAC_TIPLERI) +
    gp.quicksum(alinan_risk[r, t] * risk_ceza(r) for r in ROTALAR for t in TARIHLER),
    GRB.MINIMIZE
)

# =============================================================================
# BÖLÜM 6 — KISITLAR
# =============================================================================

for r in ROTALAR:
    kiralik_kap = KIRALIK_KAPASITE.get(r, 0)

    for t in TARIHLER:
        talep       = TALEP.get((r, t), 0)
        tampon      = risk_tamponu(r, talep, kiralik_kap)
        efektif_talep = talep + tampon
        kalan_talep = max(0, efektif_talep - kiralik_kap)

        # RİSK SINIRI: Sadece güvenlik tamponunun maksimum %25'i kadar risk alınabilir (ve en fazla 500 desi)
        maks_risk = min(tampon * 0.25, 500)
        model.addConstr(alinan_risk[r, t] <= maks_risk, name=f"maks_risk_{r.replace('→','_')}_{str(t)[:10]}")

        if kalan_talep > 0:
            model.addConstr(
                gp.quicksum(x[r, t, v] * KAPASITE[v] for v in ARAC_TIPLERI) + alinan_risk[r, t] >= kalan_talep,
                name=f"talep_{r.replace('→','_')}_{str(t)[:10]}"
            )
        else:
            model.addConstr(alinan_risk[r, t] == 0)
            for v in ARAC_TIPLERI:
                model.addConstr(x[r, t, v] == 0)

# =============================================================================
# BÖLÜM 7 & 8 — ÇÖZÜM VE RAPORLAMA
# =============================================================================

print("  Gurobi optimizer çalışıyor...")
model.optimize()

sonuclar = []
risk_kurtarilan_butce = 0

for r in ROTALAR:
    kiralik_kap = KIRALIK_KAPASITE.get(r, 0)
    km          = MESAFE.get(r, 0)

    for t in TARIHLER:
        talep       = TALEP.get((r, t), 0)
        tampon      = risk_tamponu(r, talep, kiralik_kap)
        efektif     = talep + tampon
        kalan       = max(0, efektif - kiralik_kap)

        arac_detay = {}
        spot_kap_toplam = 0
        gun_maliyet = 0

        for v in ARAC_TIPLERI:
            adet = int(round(x[r, t, v].X))
            arac_detay[v] = adet
            spot_kap_toplam += adet * KAPASITE[v]
            gun_maliyet     += adet * spot_maliyet(r, v)

        toplam_arac = sum(arac_detay.values())
        hesaplanan_risk = round(alinan_risk[r, t].X)
        mae= MAE_MAP.get(r, 0)
        boslik = kiralik_kap + spot_kap_toplam - talep

        if hesaplanan_risk > 0:
            risk_seviye = f"⚠️ Risk Alındı ({hesaplanan_risk} desi)"
            risk_kurtarilan_butce += spot_maliyet(r, "Kamyonet")
        elif mae > 0 and boslik < mae:
            risk_seviye = "🔴 Yüksek"
        elif mae > 0 and boslik < mae * 2:
            risk_seviye = "🟠 Sınırda"
        else:
            risk_seviye = "🟢 Güvenli"

        sonuclar.append({
            "Tarih"               : str(t)[:10],
            "Gün"                 : pd.Timestamp(t).strftime("%A"),
            "Rota"                : r,
            "Talep (desi)"        : round(talep),
            "Güvenlik Tamponu"    : round(tampon),
            "Alınan Risk (desi)"  : hesaplanan_risk,
            "Efektif Talep"       : round(efektif),
            "Kiralık Kapasite"    : round(kiralik_kap),
            "Kalan Talep"         : round(kalan),
            "Spot Tır"            : arac_detay.get("Tır", 0),
            "Spot Kamyon"         : arac_detay.get("Kamyon", 0),
            "Spot Hafif Kamyon"   : arac_detay.get("Hafif Kamyon", 0),
            "Spot Kamyonet"       : arac_detay.get("Kamyonet", 0),
            "Toplam Spot Araç"    : toplam_arac,
            "Mesafe (km)"         : round(km, 0),
            "Spot Maliyet (₺)"    : round(gun_maliyet),
            "Operasyonel Risk"    : risk_seviye,
        })

df_sonuc = pd.DataFrame(sonuclar)

# =============================================================================
# BÖLÜM 9 & 10 — ÖZET VE KAYDET
# =============================================================================

toplam_talep   = df_sonuc["Talep (desi)"].sum()
toplam_kiralik = df_sonuc["Kiralık Kapasite"].sum()
toplam_kalan   = df_sonuc["Kalan Talep"].sum()
toplam_arac    = df_sonuc["Toplam Spot Araç"].sum()
toplam_spot_maliyet = model.ObjVal

baseline = 0
for r in ROTALAR:
    for t in TARIHLER:
        talep = TALEP.get((r, t), 0)
        adet = int(np.ceil(talep / KAPASITE["Kamyonet"]))
        baseline += adet * spot_maliyet(r, "Kamyonet")

tasarruf      = baseline - toplam_spot_maliyet
tasarruf_oran = tasarruf / baseline * 100 if baseline > 0 else 0

os.makedirs("predictions", exist_ok=True)
df_sonuc.to_csv("predictions/optimizasyon_sonuclari.csv", index=False, encoding="utf-8-sig")

ozet = pd.DataFrame([{
    "hafta"                   : "11-17 Mayıs 2026",
    "toplam_talep_desi"       : round(toplam_talep),
    "kiralik_kapsanan_desi"   : round(toplam_kiralik),
    "spot_kapsanan_desi"      : round(toplam_kalan),
    "toplam_spot_arac"        : int(toplam_arac),
    "spot_maliyet_tl"         : round(toplam_spot_maliyet),
    "baseline_maliyet_tl"     : round(baseline),
    "tasarruf_tl"             : round(tasarruf),
    "tasarruf_oran_pct"       : round(tasarruf_oran, 1),
    "risk_kurtarilan_tl"      : round(risk_kurtarilan_butce),
}])
ozet.to_csv("predictions/ozet_metrikler.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 75)
print("  OPTİMİZASYON TAMAMLANDI")
print("=" * 75)
print(f"  Toplam spot maliyet         : ₺{toplam_spot_maliyet:>11,.0f}")
print(f"  Tasarruf                    : ₺{tasarruf:>11,.0f}  (%{tasarruf_oran:.1f})")
print(f"  Risk ile Kurtarılan Bütçe   : ₺{risk_kurtarilan_butce:>11,.0f}")
print("=" * 75)