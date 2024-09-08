import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="Multi-Page App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar menu
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Page 1", "Page 2", "Page 3", "Page 4"])

# Define the page content based on the selection
if page == "Home":
    st.title("Selamat Datang")
    st.write("Ini adalah halaman utama.")
elif page == "Page 1":
    st.title("Halaman 1")
    st.write("Ini adalah halaman pertama.")
elif page == "Page 2":
    st.title("Halaman 2")
    st.write("Ini adalah halaman kedua.")
elif page == "Page 3":
    st.title("Halaman 3")
    st.write("Ini adalah halaman ketiga.")
elif page == "Page 4":
    st.title("Halaman 4")
    st.write("Ini adalah halaman keempat.")

