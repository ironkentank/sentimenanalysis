import streamlit as st
import pandas as pd
import base64
import re
import os
import io
from datetime import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from deep_translator import GoogleTranslator
from transformers import pipeline

# inisialiasi nltk dan nlp untuk analisis sentimen dari pipeline
nltk.download('punkt')
nltk.download('stopwords')
nlp = pipeline('sentiment-analysis')

# layout streamlit
def run():
    st.title("Analisis Sentimen Ulasan Aplikasi Google Play Store")
    st.write("Upload file hasil scraping untuk dianalisis")

    # Text input for the app title
    app_title = st.text_input("Masukkan Judul", "")

    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if 'content' in df.columns:
            st.write(f"Jumlah Ulasan : {len(df)}")
            df = preprocess_data(df)

            if st.button("Analisis dalam sekali klik!"):
                if app_title:
                    df = clean_text(df, "content", "text_clean")
                    df = stopword_text(df)
                    df = tokenize_text(df)
                    df = perform_stemming(df)
                    df = transform(df, app_title)

                    st.write(df)
                    perform_naive_bayes_classification(df, app_title)
                    create_wordcloud(df['text_stemindo'], app_title)
                else:
                    st.warning("Tolong masukan judul untuk melanjutkan proses")
        else:
            st.write("File tidak memuat kolom 'content' yang akan dianalisis")
    else:
        st.write("Tolong upload file terlebih dahulu")

# Fungsi untuk memuat data
def load_data(uploaded_file):
    if uploaded_file.type == 'text/csv':
        return pd.read_csv(uploaded_file)
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        return pd.read_excel(uploaded_file)
    else:
        st.write("Format file yang tidak didukung. Silakan unggah file CSV atau Excel.")
        return pd.DataFrame()

# memanggil fungsi menghapus emoticon
def preprocess_data(df):
    df = drop_emoticon_only_rows(df, 'content')
    return df

# fungsi untuk menghapus baris yang hanya berisi emotikon
def drop_emoticon_only_rows(df, text_field):
    emoticon_pattern = r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251\s]+$'
    df['is_emoticon_only'] = df[text_field].str.match(emoticon_pattern)
    df = df[~df['is_emoticon_only']]
    df.drop(columns=['is_emoticon_only'], inplace=True)
    return df

# fungsi untuk membersihkan teks dari non-alfanumerik, link tautan, mention '@' dan konversi ke huruf kecil
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    df.dropna(subset=[new_text_field_name], inplace=True)
    return df

# Function to remove stopwords
def stopword_text(df):
    nltk_stopwords = set(stopwords.words('indonesian'))

    try:
        with open("id.stopwords.txt", "r", encoding="utf-8") as file:
            local_stopwords = set(file.read().splitlines())
        combined_stopwords = nltk_stopwords.union(local_stopwords)
    except FileNotFoundError:
        combined_stopwords = nltk_stopwords
        st.warning("File 'id.stopwords.txt' tidak ditemukan. Hanya menggunakan stopwords NLTK.")

    def process_text(text):
        if isinstance(text, str):
            return ' '.join([word for word in text.split() if word not in combined_stopwords])
        return text

    df['text_stopword'] = df['text_clean'].apply(process_text)
    return df

# Tokenisasi teks
def tokenize_text(df):
    df['text_token'] = df['text_stopword'].apply(lambda x: word_tokenize(x))
    return df

# melakukan stemming
def perform_stemming(df):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}
    for document in df['text_token']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    log_placeholder = st.empty()
    with st.spinner("Performing stemming..."):
        for idx, term in enumerate(term_dict.keys()):
            term_dict[term] = stemmed_wrapper(term)
            log_placeholder.text(f"Stemming kata/istilah {idx + 1}/{len(term_dict)}: {term} -> {term_dict[term]}")

    st.success("Stemming selesai!")

    def get_stemmed_term(document):
        return [term_dict.get(term, term) for term in document]

    df['text_stemindo'] = df['text_token'].apply(lambda x: ' '.join(get_stemmed_term(x)))
    df.dropna(subset=['text_stemindo'], inplace=True)
    return df

# transformasi data agar rapi
def transform(df, app_title):
    required_columns = ["content", "text_clean", "text_stopword", "text_token", "text_stemindo", "score"]
    df_clean = df[required_columns].copy()

    df_clean = df_clean.rename(columns={"content": "komentar", "score": "nilai"})

    neutral_indices = df_clean[df_clean['nilai'] == 3].index
    translator = GoogleTranslator(source='id', target='en')
    for idx in neutral_indices:
        translated_text = translator.translate(df_clean.loc[idx, 'komentar'])
        df_clean.loc[idx, 'text_english'] = translated_text
        sentiment = nlp(translated_text)[0]['label']
        df_clean.loc[idx, 'label'] = 'positif' if sentiment == 'POSITIVE' else 'negatif'

    df_clean['label'] = df_clean.apply(lambda x: 'negatif' if x['nilai'] <= 2 else ('positif' if x['nilai'] >= 4 else x['label']), axis=1)
    df_clean['text_token'] = df_clean['text_token'].apply(tuple)
    df_clean = df_clean.drop_duplicates()
    df_clean['text_token'] = df_clean['text_token'].apply(list)

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "preprocessed/"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{timestamp}_{app_title}_preprocessed_data.csv")
        df_clean.to_csv(output_file, index=False)
        st.success(f"Preprocessed data berhasil disimpan")

    except Exception as e:
        st.error(f"Kesalahan menyimpan data yang telah diproses sebelumnya: {e}")

    return df_clean

# Fungsi untuk menampilkan jumlah label
def display_label_count(df):
    positive_count = (df['label'] == 'positif').sum()
    negative_count = (df['label'] == 'negatif').sum()
    st.write(f"Jumlah dari ulasan positif: {positive_count}")
    st.write(f"Jumlah dari ulasan negatif: {negative_count}")

# fungsi untuk membuat wordcloud
def create_wordcloud(text_column, app_title):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text_column))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "visualisasi/"
    os.makedirs(output_dir, exist_ok=True)

    wordcloud_file = os.path.join(output_dir, f"{timestamp}_{app_title}_wordcloud.png")
    plt.savefig(wordcloud_file)
    st.success(f"Word cloud berhasil disimpan")

# Fungsi untuk melakukan klasifikasi Naive Bayes
def perform_naive_bayes_classification(df, app_title):
    display_label_count(df)
    st.write("Melakukan Klasifikasi Naive Bayes...")
    
    # mempersiapkan data untuk klasifikasi
    X = df['text_stemindo']
    y = df['label']
    
    # membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # membuat vektor data teks
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # melatih model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # menampilkan metrik
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    
    # menampilkan classification report
    st.subheader("Classification Report")
    st.text(report)
    
    # menampilkan confusion matrix
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    # menyimpan file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "sentimen/"
    output_dir2 = "visualisasi/"
    os.makedirs(output_dir, exist_ok=True)
    
    # menyimpan classification report
    report_file = os.path.join(output_dir, f"{timestamp}_{app_title}_classification_report.csv")
    with open(report_file, "w") as file:
        file.write(report)
    st.success(f"Classification report berhasil tersimpan.")
    
    # menyimpan confusion metrik
    conf_matrix_file = os.path.join(output_dir2, f"{timestamp}_{app_title}_confusion_matrix.png")
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.savefig(conf_matrix_file)
    st.success(f"Confusion matrix berhasil tersimpan.")


if __name__ == "__main__":
    run()
