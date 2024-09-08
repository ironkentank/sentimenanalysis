import streamlit as st
import pandas as pd
import os

# Fungsi untuk membaca konten file dengan penanganan encoding dan binary
def read_file(file_path):
    # Deteksi ekstensi file
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
        # Membaca file biner
        with open(file_path, 'rb') as file:
            return file.read()
    else:
        # Membaca file teks
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Coba encoding lain jika utf-8 gagal
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()

# Fungsi untuk membuat tombol unduh
def download_button(file_path, file_name):
    try:
        data = read_file(file_path)
        st.download_button(
            label=f"Unduh {file_name}",
            data=data,
            file_name=file_name
        )
    except Exception as e:
        st.error(f"Error reading file {file_name}: {e}")

# Fungsi untuk menampilkan folder dan mengatur lebar kolom
def display_folder_files(folder_path, title):
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        file_list = [file_name for file_name in files if os.path.isfile(os.path.join(folder_path, file_name))]

        if file_list:
            st.subheader(title)
            
            # Menampilkan nama file dalam tabel
            df = pd.DataFrame(file_list, columns=['File Name'])
            
            # Menyisipkan CSS untuk styling
            st.markdown("""
                <style>
                div.row_widget > div {width: 100%;}
                table {width: 100%;}
                th, td {text-align: left; padding: 8px;}
                table thead th:first-child {display: none;}
                table tbody td:first-child {display: none;}
                </style>
                """, unsafe_allow_html=True)
            
            st.dataframe(df, use_container_width=True)

            # Dropdown untuk memilih file
            selected_file = st.selectbox(f'Pilih file yang akan diunduh {title}:', ['Pilih file'] + file_list, key=title)

            # Cek apakah file dipilih
            if selected_file != 'Pilih file':
                file_path = os.path.join(folder_path, selected_file)
                download_button(file_path, selected_file)
        else:
            st.write("Folder kosong atau tidak ada file di dalamnya.")
    else:
        st.write("Path folder tidak valid.")

# Fungsi main sebagai titik masuk modul
def run():
    st.title('Penyimpanan Arsip')

    # Path folder yang akan dibaca
    base_path = os.path.dirname(__file__)
    preprocessing_folder_path = os.path.join(base_path, 'preprocessed/')
    sentiment_folder_path = os.path.join(base_path, 'sentimen/')
    scraped_folder_path = os.path.join(base_path, 'scraped/')
    visualisasi_folder_path = os.path.join(base_path, 'visualisasi/')

    # Display files 
    display_folder_files(scraped_folder_path, 'Files in Scraped Folder')
    display_folder_files(preprocessing_folder_path, 'Files in Preprocessing Folder')
    display_folder_files(sentiment_folder_path, 'Files in Sentiment Folder')
    display_folder_files(visualisasi_folder_path, 'Files in Visualisasi Folder')

if __name__ == "__main__":
    run()
