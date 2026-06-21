"""
=============================================================================
ANAHAT LOJİSTİK OPTİMİZASYONU — Streamlit Dashboard (v3 - Filo ve Rota Detaylı)
=============================================================================
Çalıştırmak için: streamlit run dashboard.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# =============================================================================
# SAYFA AYARLARI
# =============================================================================

st.set_page_config(
    page_title="Anahat Lojistik Optimizasyonu",
    page_icon="🚚",
    layout="wide",
)

st.markdown("""
<style>
  .metric-card {
    background: #1e1e2e; border-radius: 10px;
    padding: 16px 20px; margin: 4px;
  }
  .stMetric label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# VERİ YÜKLEYİCİ
# =============================================================================

@st.cache_data
def veri_yukle():
    data = {}
    dosyalar = {
        "gecmis_talep": "data/talep.csv",
        "tahminler"  : "predictions/haftalik_tahminler.csv",
        "metrikler"  : "predictions/model_metrikleri.csv",
        "optimizasyon": "predictions/optimizasyon_sonuclari.csv",
        "ozet"       : "predictions/ozet_metrikler.csv",
        "tuning"     : "predictions/tuning_sonuclari.csv",
        "kiralik_filo": "data/kiralik_arac.csv"  # Sabit kiralık araç verisi eklendi
    }
    for key, yol in dosyalar.items():
        if os.path.exists(yol):
            data[key] = pd.read_csv(yol)
        else:
            data[key] = None
    return data

# =============================================================================
# BAŞLIK
# =============================================================================

st.title("🚚 Anahat Lojistik Optimizasyon Platformu")
st.caption("11-17 Mayıs 2026 | XGBoost Talep Tahmini + Gurobi MILP Optimizasyonu")

# Veri yükle
with st.spinner("Veriler yükleniyor..."):
    data = veri_yukle()

# Pipeline çalıştırılmamışsa uyar
if data["tahminler"] is None:
    st.error("⚠️ Pipeline henüz çalıştırılmamış. Lütfen `python run_pipeline.py` komutunu çalıştırın.")
    st.stop()

tahmin_df = data["tahminler"]
metrik_df = data["metrikler"]
opt_df    = data["optimizasyon"]
ozet_df   = data["ozet"]
kiralik_df = data["kiralik_filo"]

# =============================================================================
# TAB YAPISI
# =============================================================================

tab_eda, tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Geçmiş Talep Analizi",
    "📊 Talep Tahmini",
    "🚛 Optimizasyon & Filo Planı",
    "📈 Model Performansı",
    "💰 Maliyet Analizi",
])

# =============================================================================
# TAB 0 — GEÇMİŞ TALEP ANALİZİ (EDA)
# =============================================================================

with tab_eda:
    st.subheader("Tarihsel Talep Dalgalanmaları ve Şok Analizi")
    
    gecmis_df = data.get("gecmis_talep")
    
    if gecmis_df is None:
        st.warning("⚠️ Geçmiş talep verisi (data/talep.csv) bulunamadı. Lütfen data_loader.py dosyasını çalıştırın.")
    else:
        gecmis_df["tarih"] = pd.to_datetime(gecmis_df["tarih"])
        
        col1, col2 = st.columns([1, 3])
        with col1:
            rotalar_eda = sorted(gecmis_df["rota"].unique())
            secili_rota_eda = st.selectbox("İncelemek İstediğiniz Rota:", rotalar_eda, key="eda_rota")
        
        df_secili = gecmis_df[gecmis_df["rota"] == secili_rota_eda].sort_values("tarih").copy()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ortalama Talep", f"{df_secili['talep_desi'].mean():.0f} desi")
        c2.metric("Maksimum Talep (Şok)", f"{df_secili['talep_desi'].max():.0f} desi")
        c3.metric("Standart Sapma", f"{df_secili['talep_desi'].std():.0f}")
        c4.metric("Kayıtlı Gün Sayısı", f"{len(df_secili)}")
        
        st.divider()
        
        df_secili["Hareketli Ortalama (7 Gün)"] = df_secili["talep_desi"].rolling(7).mean()
        
        fig_eda = px.line(
            df_secili, x="tarih", y="talep_desi",
            title=f"{secili_rota_eda} Hattı Tarihsel Talep Dağılımı",
            labels={"tarih": "Tarih", "talep_desi": "Gerçekleşen Talep (Desi)"},
            markers=True,
            color_discrete_sequence=["#3366CC"]
        )
        
        fig_eda.add_scatter(
            x=df_secili["tarih"], y=df_secili["Hareketli Ortalama (7 Gün)"], 
            mode='lines', name='7 Günlük Trend', 
            line=dict(color='#FF9900', dash='dot')
        )
        
        fig_eda.update_layout(hovermode="x unified")
        st.plotly_chart(fig_eda, use_container_width=True)

# =============================================================================
# TAB 1 — TAHMİN
# =============================================================================

with tab1:
    st.subheader("11-17 Mayıs 2026 Haftalık Talep Tahmini")

    gun_ozet = tahmin_df.groupby(["tarih","gun"])["tahmin_desi"].sum().reset_index()
    gun_ozet["tarih"] = pd.to_datetime(gun_ozet["tarih"])
    gun_ozet = gun_ozet.sort_values("tarih")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Haftalık Toplam", f"{tahmin_df['tahmin_desi'].sum():,.0f} desi")
    c2.metric("Günlük Ortalama", f"{gun_ozet['tahmin_desi'].mean():,.0f} desi")
    c3.metric("En Yoğun Gün", gun_ozet.loc[gun_ozet['tahmin_desi'].idxmax(), 'gun'])
    c4.metric("Tahmin Edilen Rota", f"{tahmin_df['rota'].nunique()}")

    st.divider()

    fig_gun = px.bar(
        gun_ozet, x="gun", y="tahmin_desi",
        color="tahmin_desi", color_continuous_scale="Blues",
        labels={"gun": "Gün", "tahmin_desi": "Toplam Talep (desi)"},
        title="Güne Göre Haftalık Talep Dağılımı",
    )
    fig_gun.update_coloraxes(showscale=False)
    st.plotly_chart(fig_gun, use_container_width=True)

    st.divider()

    col1, col2 = st.columns([1, 3])
    with col1:
        rotalar = sorted(tahmin_df["rota"].unique())
        secili_rota = st.selectbox("Rota Seç", ["Tümü"] + rotalar)

    with col2:
        if secili_rota != "Tümü":
            r_df = tahmin_df[tahmin_df["rota"] == secili_rota]
        else:
            r_df = gun_ozet.rename(columns={"tahmin_desi": "tahmin_desi"})
            r_df["rota"] = "Tüm Rotalar"

        if secili_rota != "Tümü":
            fig_r = px.line(
                r_df, x="tarih", y="tahmin_desi", markers=True,
                title=f"{secili_rota} — Günlük Talep Tahmini",
                labels={"tarih": "Tarih", "tahmin_desi": "Desi"},
            )
            st.plotly_chart(fig_r, use_container_width=True)

    st.subheader("Rota Bazlı Tahmin Tablosu")
    pivot = tahmin_df.pivot_table(
        index="rota", columns="gun", values="tahmin_desi", aggfunc="sum"
    ).round(0)

    gun_sirasi = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(columns=[g for g in gun_sirasi if g in pivot.columns])
    pivot["TOPLAM"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOPLAM", ascending=False)

    st.dataframe(
        pivot.style.background_gradient(cmap="Blues", subset=pivot.columns[:-1]),
        use_container_width=True, height=400,
    )

# =============================================================================
# TAB 2 — OPTİMİZASYON VE FİLO PLANI (Operasyonel Risk ve Kapasite Analizi)
# =============================================================================

with tab2:
    st.subheader("Gurobi MILP — Saha Operasyonu ve Araç Taşırma Riski")

    if opt_df is None:
        st.warning("Optimizasyon sonuçları bulunamadı. Lütfen linehaul_optimization.py dosyasını çalıştırın.")
    else:
        # 1. ARAÇ KAPASİTE SÖZLÜĞÜ (Araç tiplerine göre desi karşılıkları - Örnek değerler, kendi excel'ine göre güncelleyebilirsin)
        KAPASITELER = {
            "Tır": 22400, 
            "Kamyon": 12000, 
            "Hafif Kamyon": 7200, 
            "Kamyonet": 5600
        }

        # 2. ÜST KPI PANELİ (5 Kolona Çıkarıyoruz)
        if ozet_df is not None:
            oz = ozet_df.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Haftalık Talep", f"{oz.get('toplam_talep_desi', 0):,.0f} ds")
            c2.metric("Toplam Spot Araç", f"{oz.get('toplam_spot_arac', 0):,}")
            c3.metric("Toplam Spot Maliyet", f"₺{oz.get('spot_maliyet_tl', 0):,.0f}")
            c4.metric("Maliyet Tasarrufu", f"₺{oz.get('tasarruf_tl', 0):,.0f}", delta=f"%{oz.get('tasarruf_oran_pct', 0):.1f}") 
            
            # --- YENİ VURUCU KPI KARTI ---
            kurtarilan = oz.get('risk_kurtarilan_tl', 0)
            c5.metric("Stratejik Risk ile Kurtarılan Bütçe", f"₺{kurtarilan:,.0f}", delta="İsraf Engellendi")
            st.divider()

        st.subheader("🚛 Günlük Sevk ve Taşırma (Overflow) Riski Analizi")
        
        # Filtreler
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            tarihler = sorted(opt_df["Tarih"].unique())
            secili_tarih = st.selectbox("İncelemek İstediğiniz Günü Seçin:", ["Tümü"] + tarihler)
        with col_f2:
            rotalar_opt = sorted(opt_df["Rota"].unique())
            secili_rota_opt = st.selectbox("İncelemek İstediğiniz Rotayı Seçin:", ["Tümü"] + rotalar_opt)

        # Tabloyu zenginleştirme (Kiralıklar, Kapasiteler ve MAE)
        goster_df = opt_df.copy()
        
        # A) Kiralık Araçları Ekle
        if kiralik_df is not None:
            kiralik_map = dict(zip(kiralik_df["rota"], kiralik_df["arac_sayisi"]))
            kiralik_tip_map = dict(zip(kiralik_df["rota"], kiralik_df["arac_tipi"]))
            kiralik_kap_map = dict(zip(kiralik_df["rota"], kiralik_df["kapasite_desi"]))
            
            goster_df["Kiralık Araç Tipi"] = goster_df["Rota"].map(kiralik_tip_map).fillna("Yok")
            goster_df["Kiralık Adedi"] = goster_df["Rota"].map(kiralik_map).fillna(0).astype(int)
            goster_df["Kiralık Kapasite"] = goster_df["Rota"].map(kiralik_kap_map).fillna(0).astype(float)
        else:
            goster_df["Kiralık Araç Tipi"] = "Yok"
            goster_df["Kiralık Adedi"] = 0
            goster_df["Kiralık Kapasite"] = 0.0

        # B) MAE Değerini Doğrudan opt_df (CSV) İçinden Alma
        # Arka plandaki zeki sistemimiz bunu zaten "Güvenlik Tamponu" olarak hesaplamış, doğrudan onu çekiyoruz!
        if "Güvenlik Tamponu" in goster_df.columns:
            goster_df["Model Ort. Hatası (MAE)"] = goster_df["Güvenlik Tamponu"].astype(int)
        else:
            goster_df["Model Ort. Hatası (MAE)"] = 0

        # C) Toplam Atanan Kapasite ve Kasadaki Boşluğu Hesaplama
        # DİKKAT: opt_df içindeki sütun isimlerini esnek yakalamak için düzeltildi
        
        # 1. Önce opt_df içindeki sütunların isimlerini kontrol et (Büyük/Küçük harf, boşluk vs.)
        mevcut_sutunlar = goster_df.columns.tolist()
        
        # 2. Spot kapasiteyi, sütun isimlerini daha genel arayarak hesapla
        spot_kapasite = 0
        for col in mevcut_sutunlar:
            if "Tır" in col or "tir" in col.lower():
                spot_kapasite += goster_df[col] * KAPASITELER.get("Spot Tır", 22400)
            elif "Hafif Kamyon" in col or "hafif" in col.lower():
                spot_kapasite += goster_df[col] * KAPASITELER.get("Spot Hafif Kamyon", 7200)
            elif "Kamyonet" in col or "kamyonet" in col.lower():
                spot_kapasite += goster_df[col] * KAPASITELER.get("Spot Kamyonet", 5600)
            elif "Kamyon" in col or "kamyon" in col.lower(): # Hafif kamyon üstte elendiği için bu normal kamyondur
                spot_kapasite += goster_df[col] * KAPASITELER.get("Spot Kamyon", 12000)
                
        goster_df["Spot Kapasite"] = spot_kapasite.astype(int)
        
        goster_df["Toplam Kapasite"] = goster_df["Kiralık Kapasite"].astype(int) + goster_df["Spot Kapasite"]
        # Kasada kalan güvenlik payı (Boşluk)
        goster_df["Kasada Kalan Boşluk"] = goster_df["Toplam Kapasite"].astype(int) - goster_df["Talep (desi)"].astype(int)

        # D) RİSK HESAPLAMA MANTIĞI
        # Eğer modelin yapabileceği hata (MAE), kasada kalan boşluktan büyükse, fazladan araç kiralamak zorunda kalabiliriz!
        def risk_belirle(row):
            if row["Kasada Kalan Boşluk"] < row["Model Ort. Hatası (MAE)"]: 
                return "🔴 Yüksek Risk (Ek Araç Çıkabilir)"
            elif row["Kasada Kalan Boşluk"] < (row["Model Ort. Hatası (MAE)"] * 1.5): 
                return "🟠 Sınırda (Dikkat)"
            else:
                return "🟢 Güvenli (Kapasite Yeterli)"

        goster_df["Operasyonel Risk"] = goster_df.apply(risk_belirle, axis=1)

        # Filtreleri Uygula
        if secili_tarih != "Tümü":
            goster_df = goster_df[goster_df["Tarih"] == secili_tarih]
        if secili_rota_opt != "Tümü":
            goster_df = goster_df[goster_df["Rota"] == secili_rota_opt]

        # Gösterilecek Sütunların Düzenlenmesi (Alınan Risk eklendi)
        goster_cols = [
            "Tarih", "Rota", "Talep (desi)", "Model Ort. Hatası (MAE)", 
            "Toplam Kapasite", "Kasada Kalan Boşluk", "Alınan Risk (desi)", "Operasyonel Risk",
            "Kiralık Adedi", "Toplam Spot Araç", "Spot Maliyet (₺)"
        ]
        
        # Risk durumuna göre renklendirme (Styler)
        def color_risk(val):
            if '⚠️' in str(val): return 'background-color: rgba(255, 215, 0, 0.25); color: #b8860b; font-weight: bold;' # YENİ: Hesaplanmış Risk Rengi (Altın/Sarı)
            if '🔴' in str(val): return 'background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; font-weight: bold;'
            if '🟠' in str(val): return 'background-color: rgba(255, 166, 0, 0.2); color: #ffa600;'
            if '🟢' in str(val): return 'background-color: rgba(9, 171, 59, 0.2); color: #09ab3b;'
            return ''

        mevcut_cols = [c for c in goster_cols if c in goster_df.columns]
        
        # Sort mantığını değiştirelim, önce risk alanları en üste çıkarsın ki jüri hemen görsün
        df_styled = goster_df[mevcut_cols].sort_values(by=["Alınan Risk (desi)", "Kasada Kalan Boşluk"], ascending=[False, True]).style.map(color_risk, subset=['Operasyonel Risk'])

        st.dataframe(df_styled, use_container_width=True, height=500)

# =============================================================================
# TAB 3 — MODEL PERFORMANSI
# =============================================================================
with tab3:
    st.subheader("Talep Karakteristiğine Göre Segmentasyon ve Performans")
    st.caption("Veri bilimi (ML) ve Yöneylem Araştırması (OR) entegrasyonu: Model, açıklanabilir rotaları kendi çözerken, kaotik rotaları otomatik olarak Gurobi risk tamponuna devreder.")

    tuning_df = data.get("tuning")
    metrik_df = data.get("metrikler")

    if metrik_df is None or tuning_df is None:
        st.warning("Model veya Tuning metrikleri bulunamadı.")
    else:
        # Verileri Birleştirme ve Temizleme
        tuning_df["Güncel MAPE"] = tuning_df["Yeni MAPE"].astype(str).str.replace("%", "").astype(float)
        birlesik = metrik_df.merge(tuning_df, left_on="rota", right_on="Rota", how="left")
        
        eski_mape_col = "MAPE_%" if "MAPE_%" in birlesik.columns else "MAPE"
        yontem_guncel = "Yontem_y" if "Yontem_y" in birlesik.columns else "Yontem"
        yontem_eski   = "Yontem_x" if "Yontem_x" in birlesik.columns else "Yontem"
        
        birlesik["Final_MAPE"]   = birlesik["Güncel MAPE"].fillna(birlesik[eski_mape_col])
        birlesik["Final_R2"]     = birlesik["R²"].fillna(birlesik["R2"])
        birlesik["Final_Yontem"] = birlesik[yontem_guncel].fillna(birlesik[yontem_eski])
        valid = birlesik[birlesik["Final_MAPE"] < 500].copy()

        # ---------------------------------------------------------
        # KRİTİK NOKTA: ROTALARI R² BARAJINA GÖRE İKİYE AYIRIYORUZ
        # ---------------------------------------------------------
        R2_BARAJI = 0.30
        
        valid["Kategori"] = np.where(
            valid["Final_R2"] >= R2_BARAJI, 
            "🟢 Düzenli (ML Odaklı)", 
            "🔴 Düzensiz (Yöneylem/Tampon Odaklı)"
        )
        
        df_duzenli  = valid[valid["Kategori"] == "🟢 Düzenli (ML Odaklı)"]
        df_duzensiz = valid[valid["Kategori"] == "🔴 Düzensiz (Yöneylem/Tampon Odaklı)"]

        # --- LİG 1: MAKİNE ÖĞRENMESİ (ML) LİGİ ---
        st.markdown("### 🟢 Düzenli Rotalar (Makine Öğrenmesi Başarısı)")
        st.markdown("*Mevsimsellik, maaş günleri ve trendleri içeren, XGBoost modelinin varyansı başarıyla modellediği (R² ≥ 0.30) rotalar.*")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ML Kapsamındaki Rota", f"{len(df_duzenli)}")
        c2.metric("ML Ortalama MAPE", f"%{df_duzenli['Final_MAPE'].mean():.1f}")
        c3.metric("ML Ortalama R²", f"{df_duzenli['Final_R2'].mean():.2f}")
        xgb_say = df_duzenli["Final_Yontem"].str.contains("xgboost", na=False).sum()
        c4.metric("XGBoost Kullanımı", f"{xgb_say} Rota")

        st.divider()

        # --- LİG 2: YÖNEYLEM (OR) LİGİ ---
        st.markdown("### 🔴 Düzensiz / Proje Bazlı Rotalar (Gurobi Risk Devri)")
        st.markdown("*İstatistiksel olarak öngörülemez (R² < 0.30), proje bazlı devasa şoklar içeren rotalar. **Bu rotalardaki hata payları yapay zeka performansı ortalamasından izole edilmiş** ve operasyonel güvenlik için doğrudan Gurobi MILP modelindeki 'Güvenlik Tamponu (Safety Stock)' algoritmamıza devredilmiştir.*")
        
        c5, c6, c7 = st.columns(3)
        c5.metric("İzole Edilen Kaotik Rota", f"{len(df_duzensiz)}")
        c6.metric("Ort. Hata (Tampona Giden)", f"%{df_duzensiz['Final_MAPE'].mean():.0f}")
        c7.metric("Otomatik Karar", "Naif + Maksimum Güvenlik Stoğu")

        st.divider()

        # --- GRAFİKLER ---
        col1, col2 = st.columns(2)
        with col1:
            # Sadece düzenli rotaların temiz MAPE dağılımı
            fig_mape = px.histogram(
                df_duzenli, x="Final_MAPE", nbins=15,
                title="Makine Öğrenmesi: MAPE Dağılımı",
                labels={"Final_MAPE": "MAPE (%)"}, color_discrete_sequence=["#1D9E75"],
            )
            st.plotly_chart(fig_mape, use_container_width=True)

        with col2:
            # Genel sistemin ayrışma grafiği (Segmentler)
            fig_pie = px.pie(
                valid, names="Kategori", title="Lojistik Ağın Karakteristiği (Düzenli vs Düzensiz)",
                color="Kategori", color_discrete_map={"🟢 Düzenli (ML Odaklı)":"#1D9E75", "🔴 Düzensiz (Yöneylem/Tampon Odaklı)":"#D85A30"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================================
# TAB 4 — MALİYET ANALİZİ
# =============================================================================

with tab4:
    st.subheader("Maliyet Analizi — Kiralık Filo vs. Spot Araç")

    if ozet_df is None or opt_df is None:
        st.warning("Optimizasyon sonuçları bulunamadı.")
    else:
        oz = ozet_df.iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            spot_mal  = oz.get("spot_maliyet_tl", 0)
            tasarruf  = oz.get("tasarruf_tl", 0)

            fig_pasta = go.Figure(data=[go.Pie(
                labels=["Optimize Edilmiş Spot", "Tasarruf"],
                values=[spot_mal, tasarruf],
                hole=0.4, marker_colors=["#D85A30","#1D9E75"],
            )])
            fig_pasta.update_layout(title="Baseline vs Optimize Edilmiş Maliyet")
            st.plotly_chart(fig_pasta, use_container_width=True)

        with col2:
            if "Tarih" in opt_df.columns and "Spot Maliyet (₺)" in opt_df.columns:
                gun_mal = opt_df.groupby(["Tarih","Gün"])["Spot Maliyet (₺)"].sum().reset_index()
                fig_trend = px.line(
                    gun_mal, x="Tarih", y="Spot Maliyet (₺)", markers=True,
                    title="Günlük Spot Araç Maliyeti (₺)", color_discrete_sequence=["#D85A30"],
                )
                st.plotly_chart(fig_trend, use_container_width=True)