import streamlit as st
import nltk

# Importing page functions
from page1 import run as page1_run
from page2 import run as page2_run
from page3 import run as page3_run
from page4 import run as page4_run

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

ensure_nltk_data()

# Set up the page configuration
st.set_page_config(
    page_title="Multi-Page App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar menu
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "✍🏼 Scraping Data", "🔍 Analisis_Sentimen", "⚖️ Prediksi_Teks", "📃 Arsip"])

# Define the page content based on the selection
if page == "🏠 Home":
    st.title("Selamat Datang")
    st.write("""
    Ini adalah halaman utama aplikasi analisis sentimen Anda.
    Gunakan menu di sebelah kiri untuk mengakses fitur-fitur berikut:
    - **Analisis Sentimen**: Melakukan analisis sentimen menggunakan metode Naive Bayes.
    - **Arsip**: Melihat arsip data yang sudah dianalisis.
    - **Prediksi Teks**: Melakukan prediksi sentimen pada teks baru.
    - **Scraping Data**: Mengumpulkan data dari web untuk dianalisis.
    """)
elif page == "✍🏼 Scraping Data":
    page1_run()
elif page == "🔍 Analisis_Sentimen":
    page2_run()
elif page == "⚖️ Prediksi_Teks":
    page3_run()
elif page == "📃 Arsip":
    page4_run()
