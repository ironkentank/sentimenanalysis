import streamlit as st

def main():
    st.write("""
    Ini adalah halaman utama aplikasi analisis sentimen Anda.
    Gunakan menu di sebelah kiri untuk mengakses fitur-fitur berikut:
    - **Analisis Sentimen**: Melakukan analisis sentimen menggunakan metode Naive Bayes.
    - **Arsip**: Melihat arsip data yang sudah dianalisis.
    - **Prediksi Teks**: Melakukan prediksi sentimen pada teks baru.
    - **Scraping Data**: Mengumpulkan data dari web untuk dianalisis.
    """)

if __name__ == "__main__":
    main()
