import pandas as pd
import numpy as np

# 1. Attığın veri setini yükle (Dosya adını kendi yoluna göre düzenleyebilirsin)
df = pd.read_csv("data/talep.csv", parse_dates=["tarih"])

# 2. Test Edilecek Özellikleri (Feature) Üret
# Hipotez 1: Maaş günleri (ayın başı ve 15'i civarı) talep artar
df["maas_gunu_etkisi"] = df["tarih"].dt.day.isin([1, 2, 3, 5, 6, 7, 15, 16, 17]).astype(int)

# Hipotez 2: Mayıs ayının 11'i ve 12'si (Anneler Günü sonrası) iade dalgası yaratır
df["kampanya_iade_dalgasi"] = ((df["tarih"].dt.month == 5) & (df["tarih"].dt.day.isin([11, 12]))).astype(int)

print("=" * 60)
print("  FEATURE KORELASYON ANALİZİ (Gerçek Veri Üzerinden)")
print("=" * 60)

sonuclar = []

# 3. Her rota için bu özelliklerin gerçek taleple korelasyonuna bak
for rota in df["rota"].unique():
    df_rota = df[df["rota"] == rota].copy()
    
    # Sadece yeterli verisi olan rotaları test et
    if len(df_rota) > 50: 
        # Maaş günü ile talep arasındaki korelasyon
        corr_maas = df_rota["talep_desi"].corr(df_rota["maas_gunu_etkisi"])
        
        # Eğer Mayıs ayı verisi varsa iade dalgasını test et
        corr_iade = df_rota["talep_desi"].corr(df_rota["kampanya_iade_dalgasi"])
        
        sonuclar.append({
            "Rota": rota,
            "Maaş Günü Etkisi (r)": round(corr_maas, 3) if not np.isnan(corr_maas) else 0,
            "İade Dalgası Etkisi (r)": round(corr_iade, 3) if not np.isnan(corr_iade) else 0
        })

# 4. Sonuçları DataFrame'e çevir ve analiz et
sonuc_df = pd.DataFrame(sonuclar)

# En çok maaş günü etkisi gören ilk 10 rotayı listele
print("\n🔥 Maaş Gününden EN ÇOK Etkilenen İlk 10 Rota:")
print(sonuc_df.sort_values(by="Maaş Günü Etkisi (r)", ascending=False).head(10).to_string(index=False))

print("\n🧊 Maaş Gününden HİÇ Etkilenmeyen / Negatif Etkilenen Rotalar:")
print(sonuc_df.sort_values(by="Maaş Günü Etkisi (r)", ascending=True).head(5).to_string(index=False))

import pandas as pd
import matplotlib.pyplot as plt

# Veriyi oku ve sadece Manisa→Denizli rotasını filtrele
df = pd.read_csv("data/talep.csv", parse_dates=["tarih"])
df_md = df[df["rota"] == "Manisa→Denizli"].sort_values("tarih")

# Gerçek talep ile dümdüz ortalama çizgisini karşılaştır
plt.figure(figsize=(12, 5))
plt.plot(df_md["tarih"], df_md["talep_desi"], label="Gerçek Talep (Kaos)", color="blue", marker="o")
plt.axhline(df_md["talep_desi"].mean(), color="red", linestyle="--", label="Sıradan Ortalama")
plt.title("Manisa→Denizli: Negatif R² Teşhisi")
plt.legend()
plt.show()

# import pandas as pd
# import numpy as np

# # 1. Ham veriyi oku
# df = pd.read_csv("data/talep.csv", parse_dates=["tarih"])

# # Gözümüze batan o yüksek MAPE'li ama yüksek R²'li rotalar
# sorunlu_rotalar = ["Yalova→Mersin", "Kocaeli→Balıkesir", "Eskişehir→Balıkesir", "Tekirdağ→Isparta"]

# print("="*60)
# print("DENOMINATOR EFFECT (PAYDA YANILGISI) TEŞHİS RAPORU")
# print("="*60)

# for rota in sorunlu_rotalar:
#     df_rota = df[df["rota"] == rota]
#     if df_rota.empty: 
#         continue
        
#     # Sıfır olmayan, gerçek bir sevkiyatın olduğu günleri al
#     aktif_gunler = df_rota[df_rota["talep_desi"] > 0].copy()
    
#     # 20 desi (neredeyse 1-2 koli) ve altındaki mikro sevkiyat günleri
#     mikro_gunler = aktif_gunler[aktif_gunler["talep_desi"] <= 500]
    
#     # Ortalama talep (büyük şoklar dahil)
#     ortalama_talep = aktif_gunler["talep_desi"].mean()
    
#     print(f"\n🚀 {rota.upper()}")
#     print(f"-> Toplam aktif sevkiyat günü: {len(aktif_gunler)}")
#     print(f"-> 500 desinin altında (çok düşük) yük çıkan gün sayısı: {len(mikro_gunler)}")
#     print(f"-> Genel ortalama talep: {ortalama_talep:.0f} desi")
    
#     if len(mikro_gunler) > 0:
#         ornek = mikro_gunler.sort_values("talep_desi").iloc[0]
#         gercek_deger = ornek["talep_desi"]
        
#         # Modelin güvenli limanda kalıp (örneğin ortalama bir değer veya 50 desi) tahmin yaptığını varsayalım
#         varsayimsal_tahmin = 50.0 
#         mutlak_hata = abs(varsayimsal_tahmin - gercek_deger)
#         mape_etkisi = (mutlak_hata / gercek_deger) * 100
        
#         print(f"\n   [Matematiksel İspat - Tarih: {ornek['tarih'].date()}]")
#         print(f"   Gerçek Talep: {gercek_deger} desi (Çok düşük)")
#         print(f"   Model sadece {varsayimsal_tahmin} desi tahmin etse;")
#         print(f"   Fiziksel Sapma: {mutlak_hata} desi (Operasyonel etkisi SIFIR, 1 palet bile değil)")
#         print(f"   MAPE Etkisi: %{mape_etkisi:.0f} !!! (Sistemi çökerten yüzdelik)")
# print("\n"+"="*60)

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings("ignore")

print("="*65)
print(" 🧪 FEATURE A/B TEST SİMÜLASYONU BAŞLIYOR...")
print("="*65)

# 1. Veriyi Oku
df = pd.read_csv("data/talep.csv", parse_dates=["tarih"])

# Temel Tatil/Takvim Bilgileri (Ayar bozulmasın diye hızlıca üretiyoruz)
TATILLER_2026 = set(pd.to_datetime(["2026-01-01","2026-04-23","2026-05-01","2026-05-19","2026-08-30","2026-10-29"]))
df["tatil_mi"] = df["tarih"].isin(TATILLER_2026).astype(int)
df["haftanin_gunu"] = df["tarih"].dt.dayofweek
df["pazartesi_mi"] = (df["haftanin_gunu"] == 0).astype(int)
df["ay_basi_mu"] = (df["tarih"].dt.day <= 7).astype(int)
df["tatil_oncesi"] = df["tarih"].apply(lambda t: int((t + pd.Timedelta(1)) in TATILLER_2026))

BAZ_FEATURE_SET = ["lag_1_norm", "lag_7_norm", "roll7_mean_norm", "roll7_std_norm", "haftanin_gunu"]
YENI_FEATURE_SET = ["cv_7gun", "ivme_1_7", "lag_28_norm", "tatil_oncesi_cuma", "pazartesi_sendromu"]

TUM_FEATURELAR = BAZ_FEATURE_SET + YENI_FEATURE_SET

sonuclar = []

for rota in df["rota"].unique():
    df_rota = df[df["rota"] == rota].copy().sort_values("tarih")
    
    # -- GÜNLÜK BAZ (Standartlaştırma) --
    gun_baz = df_rota.groupby("haftanin_gunu")["talep_desi"].median().to_dict()
    df_rota["gun_baz"] = df_rota["haftanin_gunu"].map(gun_baz).clip(lower=1)
    df_rota["talep_norm"] = (df_rota["talep_desi"] / df_rota["gun_baz"]).clip(0.1, 3.0)
    s = df_rota["talep_norm"]

    # -- BAZ FEATURELAR --
    df_rota["lag_1_norm"] = s.shift(1)
    df_rota["lag_7_norm"] = s.shift(7)
    df_rota["roll7_mean_norm"] = s.shift(1).rolling(7).mean()
    df_rota["roll7_std_norm"] = s.shift(1).rolling(7).std().fillna(0)
    
    # -- YENİ DENEYSEL FEATURELAR --
    df_rota["cv_7gun"] = df_rota["roll7_std_norm"] / (df_rota["roll7_mean_norm"] + 0.001)
    df_rota["ivme_1_7"] = df_rota["lag_1_norm"] / (df_rota["roll7_mean_norm"] + 0.001)
    df_rota["lag_28_norm"] = s.shift(28)
    df_rota["tatil_oncesi_cuma"] = df_rota["tatil_oncesi"] * (df_rota["haftanin_gunu"] == 4).astype(int)
    df_rota["pazartesi_sendromu"] = df_rota["pazartesi_mi"] * df_rota["ay_basi_mu"]

    df_feat = df_rota.dropna(subset=TUM_FEATURELAR).copy()
    
    if len(df_feat) < 50: # Yeterli veri yoksa atla
        continue
        
    # -- TRAIN / TEST SPLIT (Son 14 Gün Test) --
    test_bas = df_feat["tarih"].max() - pd.Timedelta(days=14)
    tr = df_feat[df_feat["tarih"] <= test_bas]
    te = df_feat[df_feat["tarih"] > test_bas]
    
    if len(te) < 3: continue

    y_tr, y_te = tr["talep_norm"], te["talep_norm"]
    baz_te = te["gun_baz"].values
    gercek_desi = te["talep_desi"].values

    # -- MODEL A: SADECE ESKİ FEATURELAR --
    model_a = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_a.fit(tr[BAZ_FEATURE_SET], y_tr)
    pred_a = np.maximum(0.1, model_a.predict(te[BAZ_FEATURE_SET])) * baz_te
    
    r2_a = r2_score(gercek_desi, pred_a)

    # -- MODEL B: ESKİ + YENİ FEATURELAR (A/B TEST) --
    model_b = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_b.fit(tr[TUM_FEATURELAR], y_tr)
    pred_b = np.maximum(0.1, model_b.predict(te[TUM_FEATURELAR])) * baz_te
    
    r2_b = r2_score(gercek_desi, pred_b)
    
    # Değişimi kaydet
    r2_fark = r2_b - r2_a
    sonuclar.append({"rota": rota, "r2_eski": r2_a, "r2_yeni": r2_b, "fark": r2_fark})

# -- SONUÇLARI ANALİZ ET --
sonuclar_df = pd.DataFrame(sonuclar)

# Sadece R2'si mantıklı seviyede olanları al
gecerli = sonuclar_df[(sonuclar_df["r2_eski"] > 0) | (sonuclar_df["r2_yeni"] > 0)]

kazanan_yeni = len(gecerli[gecerli["fark"] > 0.02])
kazanan_eski = len(gecerli[gecerli["fark"] < -0.02])
esit = len(gecerli) - (kazanan_yeni + kazanan_eski)

print("\n📊 A/B TEST SONUÇ RAPORU")
print(f"Toplam İncelenen Rota: {len(gecerli)}")
print(f"✅ Yeni Feature'ların Kazandığı (R² Artışı > 0.02) : {kazanan_yeni} rota")
print(f"❌ Eski Feature'ların Kazandığı (R² Düşüşü > 0.02): {kazanan_eski} rota")
print(f"➖ Etki Etmediği (Nötr / Başa Baş)             : {esit} rota")

ortalama_r2_fark = gecerli['fark'].mean()
print(f"\n📈 Ortalama R² Değişimi: {ortalama_r2_fark:+.3f}")

if ortalama_r2_fark > 0.01 and kazanan_yeni > kazanan_eski:
    print("\n🎯 KARAR: Yeni feature'lar sisteme EKLENMELİDİR. Genel performansı artırıyor.")
else:
    print("\n⚠️ KARAR: Yeni feature'lar GÜRÜLTÜ yaratıyor. Eklenmemeli veya sadece SET_B'ye konulmalıdır.")
print("="*65)