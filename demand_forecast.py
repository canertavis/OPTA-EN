"""
=============================================================================
TALEP TAHMİN MODELİ — 11-17 Mayıs 2026 (v5 — Çift Feature Seti Seçimi)
=============================================================================
Strateji: Her rota için ÜÇ model karşılaştırılır ve en iyi seçilir:
  1. Mevsimsel Naif  : gün-of-week medyanı (her zaman pozitif R² garantisi)
  2. XGBoost SET_A   : Zaman serisi odaklı — lag + rolling (sade, tutarlı)
  3. XGBoost SET_B   : Zengin bağlam — SET_A + ters rota + bimodal mod + takvim

SET_A güçlü olduğunda (yüksek lag korelasyonu, düşük CV):
  → Ek feature'lar gürültü katar, SET_A kazanır

SET_B güçlü olduğunda (ters rota sinyali var, bayram etkisi, mod değişimi):
  → Ek bağlam fark yaratır, SET_B kazanır

Seçim kriteri: MAPE (tatil±1 günler ve outlier'lar hariç)
=============================================================================
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# =============================================================================
# BÖLÜM 1 — VERİ VE TATİL TANIMLARI
# =============================================================================

df = pd.read_csv("data/talep.csv", parse_dates=["tarih"])

TATILLER_2026 = set(pd.to_datetime([
    "2026-01-01",
    "2026-03-20","2026-03-21","2026-03-22","2026-03-23",
    "2026-04-23","2026-05-01","2026-05-19",
    "2026-05-27","2026-05-28","2026-05-29","2026-05-30",
    "2026-08-30","2026-10-29",
]))

ANORMAL_GUNLER = set()
for t in TATILLER_2026:
    ANORMAL_GUNLER.add(t)
    ANORMAL_GUNLER.add(t + pd.Timedelta(1))
    ANORMAL_GUNLER.add(t - pd.Timedelta(1))

BAYRAM_HAFTALARI = set()
for t in TATILLER_2026:
    BAYRAM_HAFTALARI.add((int(t.isocalendar().year), int(t.isocalendar().week)))

df["tatil_mi"]       = df["tarih"].isin(TATILLER_2026).astype(int)
df["tatil_oncesi"]   = df["tarih"].apply(lambda t: int((t + pd.Timedelta(1)) in TATILLER_2026))
df["tatil_sonrasi"]  = df["tarih"].apply(lambda t: int((t - pd.Timedelta(1)) in TATILLER_2026))
df["bayram_haftasi"] = df["tarih"].apply(
    lambda t: int((int(t.isocalendar().year), int(t.isocalendar().week)) in BAYRAM_HAFTALARI)
)

print("=" * 65)
print(f"  Veri: {len(df):,} satır, {df['rota'].nunique()} rota")
print(f"  Aralık: {df['tarih'].min().date()} → {df['tarih'].max().date()}")
print("=" * 65)

TAHMIN_TARIHLERI = pd.date_range("2026-05-11", "2026-05-17", freq="D")

# =============================================================================
# BÖLÜM 2 — İKİ FEATURE SETİ
# =============================================================================

# ── SET A: Zaman Serisi Odaklı (sade, lag + rolling) ─────────────────────────
# Yüksek lag korelasyonlu rotalarda gürültüsüz ve tutarlı performans
# lag_7 korelasyonu > 0.5 olan rotalarda genellikle bu set kazanır
FEATURES_A = [
    "lag_1_norm", "lag_3_norm", "lag_7_norm", "lag_14_norm",
    "roll7_mean_norm", "roll7_std_norm", "roll14_mean_norm",
    "ewm7_norm",
    "haftanin_gunu", "ay", "gun_no",
    "ay_sin", "ay_cos", "gun_sin", "gun_cos",
    "hafta_sonu", "tatil_mi", "tatil_oncesi", "tatil_sonrasi",
    "bayram_haftasi",
]

# ── SET B: Zengin Bağlam (SET_A + ters rota + mod + takvim detayı) ──────────
# Ters rota sinyali güçlü (r > 0.5), bimodal davranış veya
# güçlü takvim örüntüsü olan rotalarda bu set kazanır
FEATURES_B = [
    "lag_1_norm", "lag_3_norm", "lag_7_norm", "lag_14_norm",
    "roll7_mean_norm", "roll7_std_norm", "roll14_mean_norm",
    "ewm7_norm", "gecen_hafta_toplam_norm",
    "haftanin_gunu", "ay", "gun_no", "hafta_no",
    "is_gunu_sirasi", "pazartesi_mi",
    "ay_sin", "ay_cos", "gun_sin", "gun_cos",
    "hafta_sonu", "ay_sonu_mu", "ay_basi_mu",
    "tatil_mi", "tatil_oncesi", "tatil_sonrasi",
    "bayram_haftasi", "maas_gunu_etkisi",
    "mod_sinyal",
    "ters_rota_lag1_norm",
    "rota_kategori",
]

# =============================================================================
# BÖLÜM 3 — FEATURE ENGINEERING FONKSİYONLARI
# =============================================================================

def feature_olustur_A(df_rota):
    """SET A: Sade zaman serisi feature'ları."""
    df_rota = df_rota.sort_values("tarih").copy()
    s = df_rota["talep_norm"]

    df_rota["lag_1_norm"]       = s.shift(1)
    df_rota["lag_3_norm"]       = s.shift(3)
    df_rota["lag_7_norm"]       = s.shift(7)
    df_rota["lag_14_norm"]      = s.shift(14)
    df_rota["roll7_mean_norm"]  = s.shift(1).rolling(7).mean()
    df_rota["roll7_std_norm"]   = s.shift(1).rolling(7).std()
    df_rota["roll14_mean_norm"] = s.shift(1).rolling(14).mean()
    df_rota["ewm7_norm"]        = s.shift(1).ewm(span=7, adjust=False).mean()

    df_rota["haftanin_gunu"] = df_rota["tarih"].dt.dayofweek
    df_rota["ay"]            = df_rota["tarih"].dt.month
    df_rota["gun_no"]        = df_rota["tarih"].dt.dayofyear
    df_rota["ay_sin"]        = np.sin(2 * np.pi * df_rota["ay"] / 12)
    df_rota["ay_cos"]        = np.cos(2 * np.pi * df_rota["ay"] / 12)
    df_rota["gun_sin"]       = np.sin(2 * np.pi * df_rota["haftanin_gunu"] / 7)
    df_rota["gun_cos"]       = np.cos(2 * np.pi * df_rota["haftanin_gunu"] / 7)
    df_rota["hafta_sonu"]    = (df_rota["haftanin_gunu"] >= 5).astype(int)

    for col in ["tatil_mi","tatil_oncesi","tatil_sonrasi","bayram_haftasi"]:
        if col not in df_rota.columns:
            df_rota[col] = 0

    return df_rota


def feature_olustur_B(df_rota, rota_kategori=1, ters_rota_df=None):
    """SET B: Zengin bağlam feature'ları (SET_A + ek sinyaller)."""
    df_rota = feature_olustur_A(df_rota).copy()
    s = df_rota["talep_norm"]

    # ── SET B eklentileri ─────────────────────────────────────────────────────
    df_rota["gecen_hafta_toplam_norm"] = s.shift(1).rolling(7).sum()
    df_rota["hafta_no"]      = df_rota["tarih"].dt.isocalendar().week.astype(int)
    df_rota["is_gunu_sirasi"] = df_rota["haftanin_gunu"].apply(lambda g: g+1 if g < 5 else 0)
    df_rota["pazartesi_mi"]   = (df_rota["haftanin_gunu"] == 0).astype(int)
    df_rota["ay_basi_mu"]     = (df_rota["tarih"].dt.day <= 7).astype(int)
    df_rota["maas_gunu_etkisi"] = df_rota["tarih"].dt.day.isin(
        [1,2,3,5,6,7,15,16,17]).astype(int)

    # Bimodal mod sinyali: son 3 günün normalize ort > 0.5 → yüksek mod
    roll3 = s.shift(1).rolling(3).mean()
    df_rota["mod_sinyal"] = (roll3 > 0.5).astype(float).fillna(0.5)

    for col in ["ay_sonu_mu"]:
        if col not in df_rota.columns:
            df_rota[col] = 0

    # Ters rota sinyali
    if ters_rota_df is not None and len(ters_rota_df) > 0:
        ters_map = dict(zip(ters_rota_df["tarih"], ters_rota_df["talep_norm"]))
        df_rota["ters_rota_lag1_norm"] = (
            df_rota["tarih"].shift(1).map(ters_map).fillna(1.0)
        )
    else:
        df_rota["ters_rota_lag1_norm"] = 1.0

    df_rota["rota_kategori"] = rota_kategori

    return df_rota

# =============================================================================
# BÖLÜM 4 — YARDIMCI FONKSİYONLAR
# =============================================================================

XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "gamma": 0.05, "random_state": 42, "n_jobs": -1,
}

def mape_hesapla(y_gercek, y_pred, tarihler, esik):
    """Anormal günler ve outlier'lar hariç MAPE."""
    mask = (
        ~pd.Series(pd.to_datetime(tarihler)).isin(ANORMAL_GUNLER).values &
        (np.array(y_gercek) > esik)
    )
    if mask.sum() < 3:
        return 999.0
    return mean_absolute_percentage_error(
        np.array(y_gercek)[mask], np.array(y_pred)[mask]) * 100


def xgb_dene(features, feat_fn, df_tam, test_bas, esik, rota_kat=1, ters_rota_df=None):
    """
    Verilen feature seti ile XGBoost eğit ve test MAPE/R² döndür.
    HATA DÜZELTMESİ: Feature'lar tüm veri üzerinde oluşturulur, sonra Train/Test diye kesilir.
    """
    if feat_fn == feature_olustur_A:
        df_feat = feature_olustur_A(df_tam).dropna(subset=features)
    else:
        df_feat = feature_olustur_B(df_tam, rota_kategori=rota_kat, ters_rota_df=ters_rota_df).dropna(subset=features)

    tr_f = df_feat[df_feat["tarih"] <= test_bas]
    te_f = df_feat[(df_feat["tarih"] > test_bas) & (~df_feat["tarih"].isin(ANORMAL_GUNLER))]

    if len(te_f) < 3:
        te_f = df_feat[df_feat["tarih"] > test_bas]

    if len(tr_f) < 20 or len(te_f) < 3:
        return None, 999.0, 0.0

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(tr_f[features], tr_f["talep_norm"],
              eval_set=[(te_f[features], te_f["talep_norm"])],
              verbose=False)

    norm_pred = np.maximum(0.1, model.predict(te_f[features]))
    xgb_pred  = norm_pred * te_f["gun_baz"].values
    mape      = mape_hesapla(te_f["talep_desi"].values, xgb_pred, te_f["tarih"].values, esik)
    r2        = r2_score(te_f["talep_desi"].values, xgb_pred)

    return model, mape, r2

# =============================================================================
# BÖLÜM 5 — HER ROTA İÇİN: 3 MODEL DENE, EN İYİSİNİ SEÇ
# =============================================================================

metrikler     = []
tum_tahminler = []

print(f"\n  {'Rota':<32} {'MAPE_A':>7} {'MAPE_B':>7} {'MAPE_N':>7} {'Seçilen':<16} {'R²':>6}")
print(f"  {'-'*32} {'-'*7} {'-'*7} {'-'*7} {'-'*16} {'-'*6}")

for rota in sorted(df["rota"].unique()):

    df_rota = df[df["rota"] == rota].copy().sort_values("tarih").reset_index(drop=True)
    medyan  = df_rota["talep_desi"].median()
    esik    = medyan * 0.05

    # Rota kategorisi
    rota_kat = 0 if medyan < 3000 else 1 if medyan < 8000 else 2 if medyan < 15000 else 3

    # Ters rota verisi
    ters_rota_adi = f"{rota.split('→')[1]}→{rota.split('→')[0]}"
    ters_df_rota  = df[df["rota"] == ters_rota_adi][["tarih","talep_desi"]].copy()
    if len(ters_df_rota) > 0:
        tgb = ters_df_rota.groupby(ters_df_rota["tarih"].dt.dayofweek)["talep_desi"].median()
        ters_df_rota["gun_baz_ters"] = ters_df_rota["tarih"].dt.dayofweek.map(tgb)
        ters_df_rota["talep_norm"]   = (
            ters_df_rota["talep_desi"] / ters_df_rota["gun_baz_ters"].clip(lower=1)
        ).clip(0.1, 3.0)
    else:
        ters_df_rota = None

    # Gün bazı: IQR Winsorization + bayram haftası dışla
    normal = df_rota[
        (df_rota["tatil_mi"] == 0) &
        (df_rota["talep_desi"] >= esik) &
        (df_rota["bayram_haftasi"] == 0)
    ]
    gun_baz = {}
    for g in range(7):
        g_data = normal[normal["tarih"].dt.dayofweek == g]["talep_desi"]
        if len(g_data) >= 4:
            q1, q3 = g_data.quantile(0.25), g_data.quantile(0.75)
            iqr    = q3 - q1
            g_win  = g_data[(g_data >= q1-1.5*iqr) & (g_data <= q3+1.5*iqr)]
            gun_baz[g] = g_win.median() if len(g_win) >= 2 else g_data.median()
        elif len(g_data) > 0:
            gun_baz[g] = g_data.median()
        else:
            gun_baz[g] = medyan

    # Bimodal gün tespiti
    bimodal_bazlar = {}
    train_df = df_rota[df_rota["tarih"] <= df_rota["tarih"].max() - pd.Timedelta(days=14)]
    for g in range(7):
        gd = train_df[
            (train_df["tarih"].dt.dayofweek == g) &
            (train_df["tatil_mi"] == 0) &
            (train_df["talep_desi"] >= esik) &
            (train_df["bayram_haftasi"] == 0)
        ]["talep_desi"]
        if len(gd) < 4:
            continue
        cv = gd.std() / gd.mean()
        if cv > 0.8 and gd.min() / gd.median() < 0.15:
            esik_bm = gd.median() * 0.30
            yuksek = gd[gd >= esik_bm]
            dusuk  = gd[gd <  esik_bm]
            if len(yuksek) >= 2 and len(dusuk) >= 2:
                bimodal_bazlar[g] = {"yuksek": yuksek.median()}
                gun_baz[g] = yuksek.median()

    # Eğitim/test
    test_bas  = df_rota["tarih"].max() - pd.Timedelta(days=14)
    tr_normal = df_rota[
        (df_rota["tarih"] <= test_bas) &
        (df_rota["tatil_mi"] == 0) &
        (df_rota["talep_desi"] >= esik) &
        (df_rota["bayram_haftasi"] == 0)
    ]
    tr_gun_baz = {}
    for g in range(7):
        if g in bimodal_bazlar:
            tr_gun_baz[g] = bimodal_bazlar[g]["yuksek"]
        else:
            gd = tr_normal[tr_normal["tarih"].dt.dayofweek == g]["talep_desi"]
            if len(gd) >= 4:
                q1, q3 = gd.quantile(0.25), gd.quantile(0.75)
                iqr    = q3 - q1
                gw     = gd[(gd >= q1-1.5*iqr) & (gd <= q3+1.5*iqr)]
                tr_gun_baz[g] = gw.median() if len(gw) >= 2 else gd.median()
            elif len(gd) > 0:
                tr_gun_baz[g] = gd.median()
            else:
                tr_gun_baz[g] = df_rota["talep_desi"].median()

    test_df  = df_rota[df_rota["tarih"] > test_bas]
    test_temiz = test_df[~test_df["tarih"].isin(ANORMAL_GUNLER)].copy()
    if len(test_temiz) < 5:
        test_temiz = test_df

    # normalize ekle
    df_rota["gun_baz"]    = df_rota["tarih"].dt.dayofweek.map(gun_baz)
    df_rota["talep_norm"] = (df_rota["talep_desi"] / df_rota["gun_baz"].clip(lower=1)).clip(0.1, 3.0)

    train_raw = df_rota[df_rota["tarih"] <= test_bas].copy()
    test_raw  = df_rota[
        (df_rota["tarih"] > test_bas) &
        (~df_rota["tarih"].isin(ANORMAL_GUNLER))
    ].copy()
    if len(test_raw) < 3:
        test_raw = df_rota[df_rota["tarih"] > test_bas].copy()

    # ── MODEL 1: Mevsimsel Naif ───────────────────────────────────────────────
    naive_pred = test_temiz["tarih"].dt.dayofweek.map(tr_gun_baz).fillna(medyan).values
    mape_n     = mape_hesapla(test_temiz["talep_desi"].values, naive_pred,
                              test_temiz["tarih"].values, esik)
    r2_n       = r2_score(test_temiz["talep_desi"], naive_pred) if len(test_temiz) > 1 else 0

    # ── MODEL 2: XGBoost SET A ────────────────────────────────────────────────
    model_a, mape_a, r2_a = xgb_dene(
        FEATURES_A, feature_olustur_A,
        df_rota, test_bas, esik
    )

    # ── MODEL 3: XGBoost SET B ────────────────────────────────────────────────
    model_b, mape_b, r2_b = xgb_dene(
        FEATURES_B, feature_olustur_B,
        df_rota, test_bas, esik,
        rota_kat=rota_kat, ters_rota_df=ters_df_rota
    )

    # ── EN İYİSİNİ SEÇ ───────────────────────────────────────────────────────
    secimler = {
        "naive"    : (mape_n, r2_n, None,    None,          "naive"),
        "xgb_set_a": (mape_a, r2_a, model_a, FEATURES_A,    "xgb_set_a"),
        "xgb_set_b": (mape_b, r2_b, model_b, FEATURES_B,    "xgb_set_b"),
    }
    en_iyi_aday = min(secimler, key=lambda k: secimler[k][0])
    mape_sec, r2_sec, model_sec, feat_sec, yontem_sec = secimler[en_iyi_aday]

    # Baskın feature fonksiyonunu belirle
    feat_fn_sec = (feature_olustur_A if yontem_sec == "xgb_set_a"
                   else feature_olustur_B if yontem_sec == "xgb_set_b"
                   else None)

    mape_a_str = f"%{mape_a:.1f}" if mape_a < 999 else "  —"
    mape_b_str = f"%{mape_b:.1f}" if mape_b < 999 else "  —"
    mape_n_str = f"%{mape_n:.1f}" if mape_n < 999 else "  —"
    print(f"  {rota:<32} {mape_a_str:>7} {mape_b_str:>7} {mape_n_str:>7} {yontem_sec:<16} {r2_sec:>6.3f}")

    # Modeli kaydet — feature seti bilgisiyle birlikte
    model_yolu = f"models/xgb_{rota.replace('→','_').replace(' ','_')}.pkl"
    with open(model_yolu, "wb") as f:
        pickle.dump((yontem_sec, model_sec, gun_baz, rota_kat,
                     ters_rota_adi, feat_sec), f)

    # MAE hesapla
    if yontem_sec == "naive":
        mae_val = mean_absolute_error(test_temiz["talep_desi"].values, naive_pred)
    else:
        # DÜZELTME: Feature'ları sadece 14 günlük test setinde değil, tüm veride üretip sonra kesiyoruz!
        tum_feat = (feature_olustur_A(df_rota) if yontem_sec == "xgb_set_a"
                    else feature_olustur_B(df_rota, rota_kategori=rota_kat, ters_rota_df=ters_df_rota))
        
        te_feat = tum_feat[(tum_feat["tarih"] > test_bas) & (~tum_feat["tarih"].isin(ANORMAL_GUNLER))].dropna(subset=feat_sec)
        
        if len(te_feat) < 3:
            te_feat = tum_feat[tum_feat["tarih"] > test_bas].dropna(subset=feat_sec)

        if len(te_feat) > 0:
            pred_tl = np.maximum(0.1, model_sec.predict(te_feat[feat_sec])) * te_feat["gun_baz"].values
            mae_val = mean_absolute_error(te_feat["talep_desi"].values, pred_tl)
        else:
            mae_val = 0

    metrikler.append({
        "rota"      : rota,
        "egitim_gun": len(train_raw),
        "MAE"       : round(mae_val, 1),
        "MAPE_%"    : round(mape_sec, 1),
        "R2"        : round(r2_sec, 3),
        "Yontem"    : yontem_sec,
        "MAPE_A"    : round(mape_a, 1) if mape_a < 999 else None,
        "MAPE_B"    : round(mape_b, 1) if mape_b < 999 else None,
        "MAPE_Naive": round(mape_n, 1),
    })

    # ── 11-17 Mayıs Tahmini ──────────────────────────────────────────────────
    gecmis = df_rota.copy()

    for t in TAHMIN_TARIHLERI:
        tatil    = int(t in TATILLER_2026)
        tatil_on = int((t + pd.Timedelta(1)) in TATILLER_2026)
        tatil_so = int((t - pd.Timedelta(1)) in TATILLER_2026)
        bayram_h = int((int(t.isocalendar().year), int(t.isocalendar().week)) in BAYRAM_HAFTALARI)
        baz      = gun_baz.get(t.dayofweek, medyan)

        if yontem_sec == "naive":
            tahmin_desi = max(0, round(baz * (0.35 if tatil else 1)))
        else:
            yeni = pd.DataFrame([{
                "tarih": t, "rota": rota, "talep_desi": baz,
                "gun_baz": baz, "talep_norm": 1.0,
                "haftanin_gunu": t.dayofweek, "ay": t.month,
                "yil": t.year, "gun_no": t.dayofyear,
                "hafta_no": t.isocalendar().week,
                "hafta_sonu": int(t.dayofweek >= 5),
                "ay_sonu_mu": int(t.day >= 28),
                "tatil_mi": tatil, "tatil_oncesi": tatil_on,
                "tatil_sonrasi": tatil_so, "bayram_haftasi": bayram_h,
            }])
            birlesik = pd.concat([gecmis, yeni], ignore_index=True)

            if yontem_sec == "xgb_set_a":
                birlesik_feat = feature_olustur_A(birlesik)
            else:
                birlesik_feat = feature_olustur_B(birlesik,
                                                   rota_kategori=rota_kat,
                                                   ters_rota_df=ters_df_rota)

            Xy = birlesik_feat[birlesik_feat["tarih"] == t][feat_sec]

            if Xy.empty or Xy.isnull().any().any():
                nm = 1.0
            else:
                nm = max(0.1, min(float(model_sec.predict(Xy)[0]), 3.0))

            tahmin_desi = max(0, round(baz * nm * (0.35 if tatil else 1)))

            geri = yeni.copy()
            geri["talep_desi"] = tahmin_desi
            geri["talep_norm"] = nm if not tatil else 0.35
            gecmis = pd.concat([gecmis, geri], ignore_index=True)

        tum_tahminler.append({
            "rota": rota, "tarih": t.date(), "gun": t.strftime("%A"),
            "tahmin_desi": tahmin_desi, "yontem": yontem_sec,
        })

# =============================================================================
# BÖLÜM 6 — KAYDET VE ÖZET
# =============================================================================

metrik_df = pd.DataFrame(metrikler)
metrik_df.to_csv("predictions/model_metrikleri.csv", index=False, encoding="utf-8-sig")

tahmin_df = pd.DataFrame(tum_tahminler)
tahmin_df.to_csv("predictions/haftalik_tahminler.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 65)
print("  ÖZET")
print("=" * 65)
for y in ["xgb_set_a","xgb_set_b","naive"]:
    n = (metrik_df["Yontem"] == y).sum()
    print(f"  {y:<16}: {n:>3} rota")

valid = metrik_df[metrik_df["MAPE_%"] < 999]
print(f"  Ortalama MAPE  : %{valid['MAPE_%'].mean():.1f}")
print(f"  Ortalama R²    : {valid['R2'].mean():.3f}")
print(f"  R² > 0.5 rota  : {(valid['R2']>0.5).sum()} / {len(valid)}")
print(f"  Negatif R² rota: {(valid['R2']<0).sum()} / {len(valid)}")

print("\n" + "=" * 65)
print("  11-17 MAYIS 2026 TAHMİN ÖZETİ")
print("=" * 65)
ozet = tahmin_df.groupby(["tarih","gun"])["tahmin_desi"].sum().reset_index()
print(ozet.to_string(index=False))
print(f"\n  Haftalık toplam: {tahmin_df['tahmin_desi'].sum():,.0f} desi")
print(f"  → predictions/haftalik_tahminler.csv")