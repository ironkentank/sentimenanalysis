import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

def main():
    # Cek apakah PyTorch terinstal
    try:
        import torch
    except ImportError:
        st.error("PyTorch tidak ditemukan. Silakan instal PyTorch dengan menjalankan `pip install torch` di terminal Anda.")
        st.stop()

    # Inisialisasi translator dan sentiment analysis pipeline
    translator = GoogleTranslator(source='id', target='en')
    try:
        sentiment_pipeline = pipeline('sentiment-analysis')
    except ImportError as e:
        st.error(f"Gagal menginisialisasi pipeline sentiment-analysis: {str(e)}")
        st.stop()

    # Judul aplikasi
    st.title("Prediksi Sentimen Kalimat")

    # Input text dari user
    input_text = st.text_area("Masukkan kalimat dalam Bahasa Indonesia:", "")

    if st.button("Analisis Sentimen"):
        if input_text:
            try:
                # Terjemahkan kalimat ke Bahasa Inggris
                translated_text = translator.translate(input_text)
                st.write(f"Terjemahan ke Bahasa Inggris: {translated_text}")

                # Analisis sentimen menggunakan Transformers
                result = sentiment_pipeline(translated_text)[0]
                sentiment = result['label']
                score = result['score']

                if sentiment == 'POSITIVE':
                    st.success(f"Sentimen: Positif (Skor: {score:.2f})")
                else:
                    st.error(f"Sentimen: Negatif (Skor: {score:.2f})")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
        else:
            st.warning("Masukkan kalimat untuk dianalisis.")

if __name__ == "__main__":
    main()
