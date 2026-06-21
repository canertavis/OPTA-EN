"""
=============================================================================
HİPERPARAMETRE OPTİMİZASYONU — v5 (AutoML / Dinamik Feature Entegrasyonu)
=============================================================================
"""

import numpy as np
import pandas as pd
import pickle, json, os, warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

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

df["tatil_mi"]      = df["tatil_mi"] if "tatil_mi" in df.columns else df["tarih"].isin(TATILLER_2026).astype(int)
df["tatil_oncesi"]  = df["tarih"].apply(lambda t: int((t + pd.Timedelta(1)) in TATILLER_2026))
df["tatil_sonrasi"] = df["tarih"].apply(lambda t: int((t - pd.Timedelta(1)) in TATILLER_2026))

TAHMIN_TARIHLERI = pd.date_range("2026-05-11", "2026-05-17", freq="D")

if not os.path.exists("predictions/model_metrikleri.csv"):
    raise FileNotFoundError("predictions/model_metrikleri.csv bulunamadı. Önce demand_forecast.py çalıştırın.")

metrik_df = pd.read_csv("predictions/model_metrikleri.csv")
eski_mape_map = dict(zip(metrik_df["rota"], metrik_df["MAPE_%"]))
yontem_map    = dict(zip(metrik_df["rota"], metrik_df.get("Yontem", pd.Series(["xgboost"]*len(metrik_df)))))
eski_r2_map   = dict(zip(metrik_df["rota"], metrik_df["R2"]))

# =============================================================================
# BÖLÜM 2 — DİNAMİK FEATURE SETLERİ VE FONKSİYONLARI
# =============================================================================

FEATURES_A = [
    "lag_1_norm", "lag_3_norm", "lag_7_norm", "lag_14_norm",
    "roll7_mean_norm", "roll7_std_norm", "roll14_mean_norm",
    "ewm7_norm",
    "haftanin_gunu", "ay", "gun_no",
    "ay_sin", "ay_cos", "gun_sin", "gun_cos",
    "hafta_sonu", "tatil_mi", "tatil_oncesi", "tatil_sonrasi",
    "bayram_haftasi",
]

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

def feature_olustur_A(df_rota):
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
    df_rota = feature_olustur_A(df_rota).copy()
    s = df_rota["talep_norm"]

    df_rota["gecen_hafta_toplam_norm"] = s.shift(1).rolling(7).sum()
    df_rota["hafta_no"]      = df_rota["tarih"].dt.isocalendar().week.astype(int)
    df_rota["is_gunu_sirasi"] = df_rota["haftanin_gunu"].apply(lambda g: g+1 if g < 5 else 0)
    df_rota["pazartesi_mi"]   = (df_rota["haftanin_gunu"] == 0).astype(int)
    df_rota["ay_basi_mu"]     = (df_rota["tarih"].dt.day <= 7).astype(int)
    df_rota["maas_gunu_etkisi"] = df_rota["tarih"].dt.day.isin([1,2,3,5,6,7,15,16,17]).astype(int)

    roll3 = s.shift(1).rolling(3).mean()
    df_rota["mod_sinyal"] = (roll3 > 0.5).astype(float).fillna(0.5)

    for col in ["ay_sonu_mu"]:
        if col not in df_rota.columns:
            df_rota[col] = 0

    if ters_rota_df is not None and len(ters_rota_df) > 0:
        ters_map = dict(zip(ters_rota_df["tarih"], ters_rota_df["talep_norm"]))
        df_rota["ters_rota_lag1_norm"] = df_rota["tarih"].shift(1).map(ters_map).fillna(1.0)
    else:
        df_rota["ters_rota_lag1_norm"] = 1.0

    df_rota["rota_kategori"] = rota_kategori
    return df_rota

def mape_hesapla(y_gercek, y_pred, tarihler, esik):
    tarihler = pd.to_datetime(tarihler)
    mask = (~tarihler.isin(ANORMAL_GUNLER)) & (np.array(y_gercek) > esik)
    if mask.sum() < 3:
        return 999.0
    return mean_absolute_percentage_error(np.array(y_gercek)[mask], np.array(y_pred)[mask]) * 100

# =============================================================================
# BÖLÜM 3 — OPTUNA OBJECTIVE
# =============================================================================

def objective(trial, X_tr, y_tr, X_va, y_va, baz_va, tarih_va, esik):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"       : trial.suggest_int("max_depth", 3, 7),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma"           : trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 0.5, 2.0),
        "random_state": 42, "n_jobs": -1,
    }
    m = XGBRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred = np.maximum(0.1, m.predict(X_va)) * baz_va
    return mape_hesapla(y_va.values * baz_va, pred, tarih_va, esik)

# =============================================================================
# BÖLÜM 4 — HER ROTA İÇİN TUNING
# =============================================================================

N_TRIALS = 75
tum_en_iyi    = {}
ozet          = []
tum_tahminler = []
xgb_say       = 0
naive_say     = 0
atlandi_say   = 0

print("=" * 70)
print(f"  HİPERPARAMETRE OPTİMİZASYONU — {N_TRIALS} trial")
print("=" * 70)

for rota in sorted(df["rota"].unique()):

    model_yolu = f"models/xgb_{rota.replace('→','_').replace(' ','_')}.pkl"

    try:
        with open(model_yolu, "rb") as f:
            kayit = pickle.load(f)
        if isinstance(kayit, tuple) and len(kayit) == 6:
            yontem, eski_model, gun_baz, rota_kat, ters_rota_adi, feat_listesi = kayit
        elif isinstance(kayit, tuple) and len(kayit) == 5:
            yontem, eski_model, gun_baz, rota_kat, ters_rota_adi = kayit
            feat_listesi = FEATURES_A
        else:
            yontem, eski_model, gun_baz, rota_kat, ters_rota_adi, feat_listesi = "naive", None, {}, 1, "", FEATURES_A
        if not hasattr(eski_model, "predict") and yontem != "naive":
            yontem, eski_model = "naive", None
    except:
        yontem, eski_model, gun_baz, rota_kat, ters_rota_adi, feat_listesi = "naive", None, {}, 1, "", FEATURES_A

    tuning_features = feat_listesi if feat_listesi is not None else FEATURES_A

    df_rota = df[df["rota"] == rota].copy().sort_values("tarih").reset_index(drop=True)
    medyan  = df_rota["talep_desi"].median()
    esik    = medyan * 0.05

    if not gun_baz:
        normal  = df_rota[(df_rota["tatil_mi"] == 0) & (df_rota["talep_desi"] >= esik)]
        gun_baz = normal.groupby(df_rota["tarih"].dt.dayofweek)["talep_desi"].median().to_dict()
        for g in range(7):
            if g not in gun_baz:
                gun_baz[g] = medyan

    eski_mape   = eski_mape_map.get(rota, 999.0)
    rota_yontem = yontem_map.get(rota, "naive") 

    ters_df_rota = None
    if ters_rota_adi and ters_rota_adi in df["rota"].values:
        ters_df_tmp = df[df["rota"] == ters_rota_adi][["tarih","talep_desi"]].copy()
        if len(ters_df_tmp) > 0:
            ters_gun_baz = ters_df_tmp.groupby(ters_df_tmp["tarih"].dt.dayofweek)["talep_desi"].median()
            ters_df_tmp["gun_baz_ters"] = ters_df_tmp["tarih"].dt.dayofweek.map(ters_gun_baz)
            ters_df_tmp["talep_norm"]   = (ters_df_tmp["talep_desi"] / ters_df_tmp["gun_baz_ters"].clip(lower=1)).clip(0.1, 3.0)
            ters_df_rota = ters_df_tmp

    df_rota["gun_baz"]    = df_rota["tarih"].dt.dayofweek.map(gun_baz)
    df_rota["talep_norm"] = (df_rota["talep_desi"] / df_rota["gun_baz"].clip(lower=1)).clip(0.1, 3.0)

    if len(tuning_features) <= len(FEATURES_A) + 2:  
        df_feat = feature_olustur_A(df_rota).dropna(subset=tuning_features)
    else:  
        df_feat = feature_olustur_B(df_rota, rota_kategori=rota_kat, ters_rota_df=ters_df_rota).dropna(subset=tuning_features)

    if len(df_feat) < 40:
        atlandi_say += 1
        for t in TAHMIN_TARIHLERI:
            baz = gun_baz.get(t.dayofweek, medyan)
            tatil = int(t in TATILLER_2026)
            tum_tahminler.append({
                "rota": rota, "tarih": t.date(), "gun": t.strftime("%A"),
                "tahmin_desi": max(0, round(baz * (0.35 if tatil else 1))),
                "yontem": "naive"
            })
        continue

    val_bas    = df_feat["tarih"].max() - pd.Timedelta(days=14)
    tr_m       = df_feat["tarih"] <= val_bas
    va_m       = df_feat["tarih"] >  val_bas
    X_tr, y_tr = df_feat[tr_m][tuning_features], df_feat[tr_m]["talep_norm"]
    X_va, y_va = df_feat[va_m][tuning_features], df_feat[va_m]["talep_norm"]
    baz_va     = df_feat[va_m]["gun_baz"].values
    tarih_va   = df_feat[va_m]["tarih"].values

    if len(X_va) < 3:
        X_tr, y_tr = df_feat[tuning_features], df_feat["talep_norm"]
        X_va, y_va = X_tr, y_tr
        baz_va     = df_feat["gun_baz"].values
        tarih_va   = df_feat["tarih"].values

    print(f"\n  {rota} (Mevcut En İyi: %{eski_mape:.1f} - {rota_yontem})", end=" ", flush=True)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_va, y_va, baz_va, tarih_va, esik),
        n_trials=N_TRIALS, show_progress_bar=False,
    )

    yeni_mape = study.best_value
    en_iyi    = study.best_params
    en_iyi.update({"random_state": 42, "n_jobs": -1})

    tuned = XGBRegressor(**en_iyi)
    tuned.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    tp_yeni = np.maximum(0.1, tuned.predict(X_va)) * baz_va
    if len(y_va) > 1:
        r2_yeni = r2_score(y_va.values * baz_va, tp_yeni)
    else:
        r2_yeni = 0.0

    eski_r2 = eski_r2_map.get(rota, 0.0)
    
    # -- 3. ADİL KARAR MEKANİZMASI --
    sart_1 = (yeni_mape < eski_mape) and (r2_yeni >= eski_r2 - 0.05)
    
    mape_tolerans = 5.0
    r2_sicrama    = 0.10
    sart_2 = (yeni_mape <= eski_mape + mape_tolerans) and (r2_yeni >= eski_r2 + r2_sicrama) and (r2_yeni > 0.30)

    # 1. Önce Kazananı Belirle
    if sart_1 or sart_2:
        secilen_model = tuned
        sec_mape      = yeni_mape
        sec_r2        = r2_yeni
        final_yontem  = "xgboost_tuned"
        iyilesme      = f"+%{eski_mape - yeni_mape:.1f}" if sart_1 else f"-%{yeni_mape - eski_mape:.1f} (R² Kurtardı)"
    else:
        secilen_model = eski_model
        sec_mape      = eski_mape
        sec_r2        = eski_r2 
        final_yontem  = rota_yontem
        iyilesme      = "—"

    # ---------------------------------------------------------
    # 2. PAYDA YANILGISI (DENOMINATOR EFFECT) TESPİTİ
    # R² 0.35'ın üzerindeyken MAPE %70'in üzerindeyse alarm ver!
    # ---------------------------------------------------------
    is_payda_yanilgisi = (sec_mape >= 70.0) and (sec_r2 >= 0.35)
    uyari = " ⚠️ [Denominator Effect]" if is_payda_yanilgisi else ""

    # 3. Terminale Şık Bir Şekilde Yazdır
    if sart_1 or sart_2:
        xgb_say += 1
        if sart_2 and not sart_1:
            print(f"\n      🚀 STRATEJİK ZAFER: Tuned XGB Kazandı! | MAPE: %{sec_mape:.1f} (R²: {eski_r2:.3f} ➔ {sec_r2:.3f}){uyari}")
        else:
            print(f"\n      ✅ KLASİK ZAFER: Tuned XGB Kazandı! | MAPE: %{sec_mape:.1f} (R²: {sec_r2:.3f}){uyari}")
    else:
        
        if yeni_mape >= eski_mape + mape_tolerans:
            red_sebebi = f"Hata payı çok arttı (+%{yeni_mape - eski_mape:.1f})"
        else:
            red_sebebi = f"R² kaybı tolere edilemez ({eski_r2:.3f} ➔ {r2_yeni:.3f})"
            
        model_isim = "XGBoost (Set A)" if final_yontem == "xgb_set_a" else \
                     "XGBoost (Set B)" if final_yontem == "xgb_set_b" else \
                     "Mevsimsel Naif"
                     
        if "xgb" in final_yontem:
            xgb_say += 1
            print(f"\n      🛡️ KORUNDU: {model_isim} | Red: {red_sebebi} | Sabit MAPE: %{sec_mape:.1f} (R²: {sec_r2:.3f}){uyari}")
        else:
            naive_say += 1
            print(f"\n      🛡️ KORUNDU: {model_isim} | Red: {red_sebebi} | Sabit MAPE: %{sec_mape:.1f} (R²: {sec_r2:.3f}){uyari}")
            
    # 4. Modeli Güvenle Kaydet
    with open(model_yolu, "wb") as f:
        pickle.dump((final_yontem, secilen_model, gun_baz, rota_kat, ters_rota_adi, tuning_features), f)

    tum_en_iyi[rota] = en_iyi
    ozet.append({
        "Rota"      : rota,
        "Eski MAPE" : f"%{eski_mape:.1f}",
        "Yeni MAPE" : f"%{sec_mape:.1f}",
        "İyileşme"  : iyilesme,
        "R²"        : round(sec_r2, 3),
        "Yontem"    : final_yontem
    })

    # ── 11-17 Mayıs Tahmini (Dinamik Feature Seçimi ile) ─────────────────────
    gecmis = df_rota.copy()
    
    for t in TAHMIN_TARIHLERI:
        tatil    = int(t in TATILLER_2026)
        tatil_on = int((t + pd.Timedelta(1)) in TATILLER_2026)
        tatil_so = int((t - pd.Timedelta(1)) in TATILLER_2026)
        baz      = gun_baz.get(t.dayofweek, medyan)

        if final_yontem == "naive":
            td = max(0, round(baz * (0.35 if tatil else 1)))
            tum_tahminler.append({
                "rota": rota, "tarih": t.date(), "gun": t.strftime("%A"),
                "tahmin_desi": td, "yontem": "naive"
            })
            continue

        yeni = pd.DataFrame([{
            "tarih": t, "rota": rota, "talep_desi": baz,
            "gun_baz": baz, "talep_norm": 1.0,
            "haftanin_gunu": t.dayofweek, "ay": t.month,
            "yil": t.year, "gun_no": t.dayofyear,
            "hafta_no": t.isocalendar().week,
            "hafta_sonu": int(t.dayofweek >= 5),
            "ay_sonu_mu": int(t.day >= 28),
            "tatil_mi": tatil, "tatil_oncesi": tatil_on, "tatil_sonrasi": tatil_so,
        }])

        b = pd.concat([gecmis, yeni], ignore_index=True)
        
        # DÜZELTME 2 & 3: Dinamik Feature fonksiyonları ve listeleri!
        if len(tuning_features) <= len(FEATURES_A) + 2:
            b = feature_olustur_A(b)
        else:
            b = feature_olustur_B(b, rota_kategori=rota_kat, ters_rota_df=ters_df_rota)
            
        Xy = b[b["tarih"] == t][tuning_features]

        if (Xy.empty or Xy.isnull().any().any()):
            nm = 1.0 
        else:
             nm = max(0.1, min(float(secilen_model.predict(Xy)[0]), 3.0))

        td = max(0, round(baz * nm * (0.35 if tatil else 1)))
        
        tum_tahminler.append({
            "rota": rota, "tarih": t.date(), "gun": t.strftime("%A"),
            "tahmin_desi": td, "yontem": final_yontem,
        })
        
        geri = yeni.copy()
        geri["talep_desi"] = td
        geri["talep_norm"] = nm
        gecmis = pd.concat([gecmis, geri], ignore_index=True)

# =============================================================================
# BÖLÜM 5 — KAYDET VE RAPORLA
# =============================================================================

with open("models/best_params.json","w",encoding="utf-8") as f:
    json.dump(tum_en_iyi, f, indent=2, ensure_ascii=False)

ozet_df = pd.DataFrame(ozet)
ozet_df.to_csv("predictions/tuning_sonuclari.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(tum_tahminler).to_csv("predictions/haftalik_tahminler.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("  TUNING SONUÇLARI")
print("=" * 70)
if len(ozet_df):
    print(ozet_df.to_string(index=False))
    yn = ozet_df["Yeni MAPE"].str.replace("%","").astype(float)
    print(f"\n  Tune edilen XGBoost rota : {xgb_say}")
    print(f"  Naif (atlandı) rota      : {naive_say}")
    print(f"  Yetersiz veri (atlandı)  : {atlandi_say}")
    print(f"  Ortalama MAPE (tuned)    : %{yn.mean():.1f}")
    r2_valid = ozet_df[ozet_df["R²"] > -10]["R²"]
    print(f"  R²>0.3 rota              : {(r2_valid>0.3).sum()} / {len(r2_valid)}")

print(f"\n  → predictions/haftalik_tahminler.csv güncellendi")
print(f"  Sonraki adım: python linehaul_optimization.py")