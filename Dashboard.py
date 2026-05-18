import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import itertools

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Dashboard Analitik Diabetes", page_icon="📊", layout="wide")

# CACHING DATA (MEMPERCEPAT LOAD)
@st.cache_data
def load_data():
    # 1. Load dataset asli
    df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
    
    # 2. Cleaning Data
    df.drop_duplicates(inplace=True)
    
    # 3. Deteksi & Filter Outlier BMI dengan IQR
    Q1 = df['BMI'].quantile(0.25)
    Q3 = df['BMI'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.15 * IQR
    upper_bound = Q3 + 1.15 * IQR
    df_clean = df[(df['BMI'] >= lower_bound) & (df['BMI'] <= upper_bound)].copy()
    
    # 4. Konversi Tipe Data
    bool_cols = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 
                 'NoDocbcCost', 'DiffWalk']
    df_clean[bool_cols] = df_clean[bool_cols].astype(bool)
    
    return df, df_clean

# Panggil fungsi load data
df_raw, df_clean = load_data()

# HEADER & DESKRIPSI
st.title("📊 Laporan Analitik: Faktor Risiko Diabetes (BRFSS 2015)")
st.markdown("""
Dashboard ini menyajikan hasil eksplorasi data komprehensif untuk memahami faktor-faktor utama yang membedakan 
individu sehat, prediabetes, dan diabetes. Analisis ini dibagi menjadi tiga pilar utama: 
**Gaya Hidup**, **Faktor Sosioekonomi**, dan **Riwayat Komorbiditas**.
""")
st.markdown("---")

# KONTEN DASHBOARD (TABS)
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Gambaran Umum Data", 
    "🏃‍♂️ Q1: Faktor Gaya Hidup", 
    "💰 Q2: Kesenjangan Sosioekonomi", 
    "❤️ Q3: Risiko Komorbiditas"
])

# TAB 1: OVERVIEW
with tab1:
    st.header("Gambaran Umum & Korelasi Fitur")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Distribusi Kelas Target")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(x='Diabetes_012', data=df_raw, ax=ax1, palette=['#639922', '#BA7517', '#E24B4A'])
        ax1.set_title('Distribusi Kelas (0 = Sehat, 1 = Prediabetes, 2 = Diabetes)', fontweight='bold')
        st.pyplot(fig1)
        
        st.info("**Insight Ketidakseimbangan Data:**\nDari bar chart distribusi kelas terlihat ketidakseimbangan data yang signifikan, di mana kelas Sehat mendominasi dengan sekitar 84% data, sementara Prediabetes hanya ~2% dan Diabetes ~14%. Kondisi ini memerlukan penanganan khusus (*balancing*) saat masuk ke tahap pemodelan *Machine Learning* nantinya.")
        
    with col2:
        st.subheader("Heatmap Korelasi")
        fig2, ax2 = plt.subplots(figsize=(10, 7.5))
        corr = df_raw[['Diabetes_012', 'HighBP', 'HighChol', 'BMI', 'Age', 'GenHlth', 'PhysActivity']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
        ax2.set_title('Heatmap Korelasi Fitur Kesehatan Utama', fontweight='bold')
        st.pyplot(fig2)
        
        st.info("**Insight Analisis Korelasi:**\nBerdasarkan Heatmap di atas, fitur yang paling berpengaruh kuat secara linear terhadap prediksi diabetes (memiliki nilai korelasi positif terbesar terhadap target) adalah **HighBP (Hipertensi)**, **HighChol (Kolesterol Tinggi)**, **BMI**, dan **GenHlth (Persepsi Kesehatan Umum)**.")

# TAB 2: GAYA HIDUP
with tab2:
    st.header("Q1: Faktor Gaya Hidup Pemisah Kelas Diabetes")
    st.info("💡 **Pertanyaan Bisnis:** Faktor gaya hidup mana (BMI, aktivitas fisik, konsumsi buah & sayur, alkohol, merokok) yang paling signifikan membedakan individu yang menderita diabetes dari yang tidak, berdasarkan data survei kesehatan BRFSS 2015?")
    
    # Aggregasi Data
    lifestyle_features = [
        ("BMI", "Rata-rata BMI", False),
        ("PhysActivity", "% Aktif Fisik", True),
        ("Fruits", "% Konsumsi Buah", True),
        ("Veggies", "% Konsumsi Sayur", True),
        ("HvyAlcoholConsump", "% Alkohol Berat", True),
        ("Smoker", "% Perokok", True),
    ]
    
    lifestyle_agg = (
        df_clean.groupby("Diabetes_012")[[col for col, _, _ in lifestyle_features]]
        .mean()
        .rename(index={0.0: "Sehat", 1.0: "Prediabetes", 2.0: "Diabetes"})
    )
    
    # Visualisasi
    fig_ls, axes_ls = plt.subplots(2, 3, figsize=(15, 9))
    axes_ls = axes_ls.flatten()
    class_labels = ["Sehat", "Prediabetes", "Diabetes"]
    bar_colors = ["#639922", "#BA7517", "#E24B4A"]

    for idx, (col, title, is_pct) in enumerate(lifestyle_features):
        vals = lifestyle_agg[col].values
        bars = axes_ls[idx].bar(class_labels, vals, color=bar_colors, edgecolor="white", width=0.6)
        
        for bar, v in zip(bars, vals):
            label = f"{v*100:.1f}%" if is_pct else f"{v:.2f}"
            axes_ls[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.003 if is_pct else 0.1),
                              label, ha="center", va="bottom", fontsize=10, fontweight="bold")
            
        axes_ls[idx].set_title(title, fontsize=12, fontweight="bold", pad=10)
        axes_ls[idx].spines["top"].set_visible(False)
        axes_ls[idx].spines["right"].set_visible(False)
        
        if is_pct:
            axes_ls[idx].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    # Legend Custom
    legend_el = [
        mpatches.Patch(color="#639922", label="Sehat (0)"),
        mpatches.Patch(color="#BA7517", label="Prediabetes (1)"),
        mpatches.Patch(color="#E24B4A", label="Diabetes (2)"),
    ]
    fig_ls.legend(handles=legend_el, loc="lower center", ncol=3, fontsize=11, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    st.pyplot(fig_ls)
    
    # Insight Panel
    st.success("""
    ### 📝 Kesimpulan Utama: Gaya Hidup
    1. **BMI adalah pembeda terkuat:** Penderita diabetes memiliki rata-rata BMI 5+ poin lebih tinggi dari kelompok sehat (konsisten naik di setiap kelas).
    2. **Aktivitas fisik menunjukkan pola terbalik:** Kelompok sehat paling aktif (76%), sedangkan kelompok diabetes paling rendah (64%).
    3. **Konsumsi Buah & Sayur:** Perbedaannya sangat kecil antar kelompok (~4%), sehingga ini bukan faktor pembeda utama.
    4. **Paradoks Alkohol & Merokok:** Tidak menunjukkan pola konsisten. Hal ini kemungkinan besar terjadi karena penderita diabetes sudah memodifikasi dan memperbaiki perilakunya (berhenti merokok/minum alkohol) setelah mendapatkan diagnosis medis.
    """)

# TAB 3: SOSIOEKONOMI
with tab3:
    st.header("Q2: Pengaruh Pendapatan & Akses Layanan Kesehatan")
    st.info("💡 **Pertanyaan Bisnis:** Seberapa besar pengaruh tingkat pendapatan (Income) dan tingkat pendidikan (Education) terhadap kemampuan individu mengakses layanan kesehatan (AnyHealthcare & NoDocbcCost), dan apakah kelompok berpendapatan rendah memiliki risiko diabetes lebih tinggi berdasarkan data BRFSS 2015?")
    
    # Aggregasi Data
    income_health = df_clean.groupby('Income')[['AnyHealthcare', 'NoDocbcCost']].mean().reset_index()
    income_diab = df_clean.groupby('Income').apply(lambda x: (x['Diabetes_012'] == 2.0).sum() / len(x)).reset_index(name='diabetes_pct')
    income_summary = income_health.merge(income_diab, on='Income')
    
    # Visualisasi
    fig_eco, axes_eco = plt.subplots(1, 3, figsize=(18, 5.5))
    income_labels = [str(int(i)) for i in income_summary['Income']]
    bar_income = ['#E24B4A','#E24B4A','#E05A30','#BA7517','#BA7517','#639922','#639922','#639922']

    # Panel 1: Asuransi
    bars1 = axes_eco[0].bar(income_labels, income_summary['AnyHealthcare'] * 100, color='#378ADD', edgecolor='white', width=0.7)
    for bar, (_, row) in zip(bars1, income_summary.iterrows()):
        axes_eco[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{row['AnyHealthcare']*100:.0f}%", ha='center', fontsize=9, fontweight='bold')
    axes_eco[0].set_title('Kepemilikan Asuransi Kesehatan\nper Level Pendapatan', fontsize=12, fontweight='bold')
    axes_eco[0].set_ylim(80, 102)
    axes_eco[0].spines[['top', 'right']].set_visible(False)

    # Panel 2: Hambatan Biaya Dokter
    bars2 = axes_eco[1].bar(income_labels, income_summary['NoDocbcCost'] * 100, color=bar_income, edgecolor='white', width=0.7)
    for bar, (_, row) in zip(bars2, income_summary.iterrows()):
        axes_eco[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{row['NoDocbcCost']*100:.1f}%", ha='center', fontsize=9, fontweight='bold')
    axes_eco[1].set_title('Tidak Bisa ke Dokter karena Biaya\nper Level Pendapatan', fontsize=12, fontweight='bold')
    axes_eco[1].spines[['top', 'right']].set_visible(False)

    # Panel 3: Dual Axis (Diabetes vs Asuransi)
    ax3 = axes_eco[2]
    x_idx = range(len(income_summary))
    ax3.plot(x_idx, income_summary['diabetes_pct'] * 100, color='#E24B4A', linewidth=2.5, marker='o', markersize=7, label='% Diabetes', markerfacecolor='white')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x_idx, income_summary['AnyHealthcare'] * 100, color='#378ADD', linewidth=2.5, marker='s', linestyle='--', markersize=7, label='% Asuransi', markerfacecolor='white')
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(income_labels)
    ax3.set_title('Diabetes vs Asuransi per Level Pendapatan', fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3_twin.spines['top'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig_eco)
    
    # Insight Panel
    st.success("""
    ### 📝 Kesimpulan Utama: Sosioekonomi
    1. **Prevalensi Berbanding Terbalik dengan Pendapatan:** Prevalensi diabetes pada kelompok income rendah (Level 1 = 22.0%) hampir **2.9x lebih tinggi** dibandingkan kelompok income tertinggi (Level 8 = 7.5%).
    2. **Ketimpangan Akses Layanan Medis:** Hambatan biaya untuk mengunjungi dokter sangat timpang. Terdapat **21.1%** populasi pada kelompok income rendah yang tidak bisa ke dokter karena biaya, dibandingkan hanya **3.0%** pada kelompok income tinggi.
    3. **Tingkat Pendidikan (Data Agregasi):** Semakin tinggi tingkat pendidikan, prevalensi diabetes semakin menurun secara drastis (dari 28.0% pada lulusan SD, turun ke angka 9.0% pada lulusan S2/S3).
    """)

# TAB 4: KOMORBIDITAS
with tab4:
    st.header("Q3: Dampak Kumulatif Komorbiditas (Hipertensi, Kolesterol, Jantung)")
    st.info("💡 **Pertanyaan Bisnis:** Bagaimana kombinasi hipertensi (HighBP), kolesterol tinggi (HighChol), dan riwayat penyakit jantung (HeartDiseaseorAttack) memengaruhi probabilitas seseorang terkena diabetes berdasarkan data BRFSS 2015?")
    
    # Kalkulasi Kombinasi Komorbiditas
    baseline = (df_clean['Diabetes_012'] == 2.0).mean() * 100
    results = []
    
    for combo in itertools.product([0, 1], [0, 1], [0, 1]):
        mask = (
            (df_clean['HighBP'] == combo[0]) & 
            (df_clean['HighChol'] == combo[1]) & 
            (df_clean['HeartDiseaseorAttack'] == combo[2])
        )
        pct = (df_clean[mask]['Diabetes_012'] == 2.0).mean() * 100
        results.append({
            'HighBP': combo[0], 'HighChol': combo[1], 'Heart': combo[2], 
            'diabetes_pct': pct, 'multiplier': pct / baseline
        })
    
    results_sorted = sorted(results, key=lambda x: x['diabetes_pct'], reverse=True)
    
    def get_color(pct):
        if pct >= 30: return '#E24B4A'
        if pct >= 20: return '#D85A30'
        if pct >= 12: return '#BA7517'
        if pct >= 8:  return '#639922'
        return '#3B6D11'

    fig_kom, axes_kom = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Kombinasi Horizontal Bar
    labels_combo = ['BP+Chol+Jantung', 'BP+Jantung', 'BP+Chol', 'Chol+Jantung', 'Hanya Jantung', 'Hanya BP', 'Hanya Chol', 'Tanpa Komorbiditas']
    vals_combo = [r['diabetes_pct'] for r in results_sorted]
    colors_combo = [get_color(v) for v in vals_combo]

    bars_kom1 = axes_kom[0].barh(labels_combo[::-1], vals_combo[::-1], color=colors_combo[::-1], edgecolor='white', height=0.65)
    for bar, v in zip(bars_kom1, vals_combo[::-1]):
        axes_kom[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
    axes_kom[0].set_title('Prevalensi Diabetes per Kombinasi Penyakit', fontsize=12, fontweight='bold')
    axes_kom[0].spines[['top', 'right']].set_visible(False)

    # Panel 2: Faktor Tunggal Grouped Bar
    nama_list = ['Hipertensi', 'Kolesterol Tinggi', 'Peny. Jantung']
    col_list = ['HighBP', 'HighChol', 'HeartDiseaseorAttack']
    vals_ada = [(df_clean[df_clean[c]==1.0]['Diabetes_012']==2.0).mean()*100 for c in col_list]
    vals_tdk = [(df_clean[df_clean[c]==0.0]['Diabetes_012']==2.0).mean()*100 for c in col_list]
    
    x_idx2 = np.arange(len(nama_list))
    axes_kom[1].bar(x_idx2 - 0.2, vals_ada, 0.4, color='#E24B4A', label='Ada', edgecolor='white')
    axes_kom[1].bar(x_idx2 + 0.2, vals_tdk, 0.4, color='#639922', label='Tidak', edgecolor='white')
    
    for i in range(len(nama_list)):
        axes_kom[1].text(x_idx2[i] - 0.2, vals_ada[i] + 0.5, f'{vals_ada[i]:.1f}%', ha='center', fontsize=9, fontweight='bold')
        axes_kom[1].text(x_idx2[i] + 0.2, vals_tdk[i] + 0.5, f'{vals_tdk[i]:.1f}%', ha='center', fontsize=9, fontweight='bold')
        
    axes_kom[1].set_xticks(x_idx2)
    axes_kom[1].set_xticklabels(nama_list)
    axes_kom[1].legend(frameon=False)
    axes_kom[1].set_title('Pengaruh per Faktor Tunggal', fontsize=12, fontweight='bold')
    axes_kom[1].spines[['top', 'right']].set_visible(False)

    # Panel 3: Multiplier
    mults = [r['multiplier'] for r in results_sorted]
    bars_kom3 = axes_kom[2].barh(labels_combo[::-1], mults[::-1], color=colors_combo[::-1], edgecolor='white', height=0.65)
    for bar, v in zip(bars_kom3, mults[::-1]):
        axes_kom[2].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{v:.1f}x', va='center', fontsize=9, fontweight='bold')
    axes_kom[2].axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    axes_kom[2].set_title('Multiplier Risiko (Baseline = 1x)', fontsize=12, fontweight='bold')
    axes_kom[2].spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig_kom)
    
    # Insight Panel
    st.success("""
    ### 📝 Kesimpulan Utama: Komorbiditas
    1. **Baseline Risiko Sangat Rendah:** Individu tanpa komorbiditas apa pun (Hipertensi, Kolesterol Tinggi, atau Riwayat Penyakit Jantung) hanya memiliki angka prevalensi diabetes sebesar **4.3%**.
    2. **Penyakit Jantung sebagai Faktor Tunggal Terkuat:** Secara individual, memiliki riwayat penyakit jantung menaikkan prevalensi diabetes ke angka 31.8%, dampak ini lebih besar dibandingkan hanya memiliki hipertensi (24.1%) atau kolesterol tinggi (21.9%).
    3. **Efek Multiplikatif (Bukan Sekadar Aditif):** Kombinasi ketiga komorbiditas tersebut secara bersamaan akan melesatkan risiko seseorang terkena diabetes hingga **37.9%**. Angka ini merepresentasikan peningkatan risiko sebesar **hampir 9x lipat** jika dibandingkan dengan kelompok basis tanpa komorbiditas.
    """)