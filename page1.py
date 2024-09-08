import streamlit as st
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from google_play_scraper import Sort, reviews

def scrape_reviews(app_packages, count):
    app_reviews = []
    for ap in tqdm(app_packages):
        for score in range(1, 6):
            for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
                rvs, _ = reviews(
                    ap,
                    lang='id',
                    country='id',
                    sort=sort_order,
                    count=count,
                    filter_score_with=score
                )
                for r in rvs:
                    r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
                    r['appId'] = ap
                app_reviews.extend(rvs)
    return pd.DataFrame(app_reviews)

def run():
    st.title("Scraping Ulasan Aplikasi di Google Play Store")
    
    package_name = st.text_input("Masukkan nama package aplikasi dari Play Store:")
    count_value = st.text_input("Masukkan nilai count:")
    max_display = st.text_input("Jumlah maksimal data yang ditampilkan:")

    try:
        count_value = int(count_value) if count_value.isdigit() else 2000
        max_display = int(max_display) if max_display.isdigit() else 100
    except ValueError:
        st.error("Masukkan nilai numerik yang valid.")
        return
    
    if st.button("Scrape"):
        if package_name:
            with st.spinner("Sedang melakukan scraping..."):
                try:
                    app_reviews_df = scrape_reviews([package_name], count_value)
                    
                    # Limit the DataFrame to max_display rows
                    app_reviews_df = app_reviews_df.head(max_display)
                    
                    st.write("Selesai!")
                    st.dataframe(app_reviews_df)

                    # Automatically save the DataFrame to a CSV file with a timestamp and package name
                    output_dir = "menu/pages/hasil/scraped/"
                    os.makedirs(output_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(output_dir, f"{timestamp}_{package_name}_scraped.csv")
                    
                    app_reviews_df.to_csv(output_file, index=False)
                    st.success(f"Scraped data berhasil disimpan")

                except Exception as e:
                    st.error(f"Terjadi kesalahan : {e}")
        else:
            st.warning("Mohon masukkan nama package aplikasi di Play Store.")

if __name__ == "__main__":
    run()
