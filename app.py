import streamlit as st

# Importing page functions
from page1 import run as page1_run
from page2 import run as page2_run
from page3 import run as page3_run
from page4 import run as page4_run
from page5 import run as page5_run

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
    page1_run()
elif page == "Page 2":
    page2_run()
elif page == "Page 3":
    page3_run()
elif page == "Page 4":
    page4_run()
elif page == "Page 5":
    page5_run()
