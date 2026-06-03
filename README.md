# DiaLens Dashboard — Deteksi & Analisis Risiko Diabetes (BRFSS 2015)

Proyek ini merupakan dashboard dari Capstone Project Analisis Faktor Risiko Diabetes. Aplikasi ini dibangun menggunakan **Streamlit** untuk memberikan pengalaman interaktif kepada pengguna dalam mengeksplorasi wawasan data (EDA).

## Deskripsi Singkat Fitur
**Visualisasi Data Interaktif**: Menampilkan grafik dan wawasan utama mengenai Gaya Hidup, Sosial-Ekonomi, dan Komorbiditas berdasarkan data BRFSS 2015.

## Petunjuk Setup Environment
Prasyarat
- Python 3.8 atau lebih baru
- pip
- Terminal (Bash/Zsh)

## Langkah-langkah Instalasi
1. Clone repositori
   git clone [https://github.com/CC26-JAYA/Dashboard-DiaLens.git](https://github.com/CC26-JAYA/Dashboard-DiaLens.git)
   cd dialens-dashboard

2. Buat virtual environment
   python3 -m venv venv
   source venv/bin/activate

3. Install dependensi
   pip install -r requirements.txt

4. Siapkan Model dan Dataset
   Pastikan file model terlatih (xgb_model.pkl) sudah berada di direktori utama.
   Letakkan file dataset hasil pemrosesan (jika diperlukan untuk visualisasi tambahan) ke dalam folder data/.

## Cara Menjalankan Aplikasi
   Jalankan server Streamlit lokal menggunakan perintah berikut di terminal:
   - streamlit run app.py
