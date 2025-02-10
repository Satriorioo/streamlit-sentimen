import streamlit as st
import pandas as pd
import os

# Membaca daftar kata positif dan negatif
def load_lexicon(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return set(word.strip() for word in file.readlines())

positif_words = load_lexicon("positive.txt")
negatif_words = load_lexicon("negative.txt")

# Fungsi analisis sentimen berbasis lexicon
def analyze_sentiment(text):
    words = text.lower().split()
    positif_count = sum(1 for word in words if word in positif_words)
    negatif_count = sum(1 for word in words if word in negatif_words)
    
    if positif_count > negatif_count:
        return "Positif"
    elif negatif_count > positif_count:
        return "Negatif"
    else:
        return "Netral"

# Fungsi utama untuk ditampilkan di dashboard
def show():
    st.title("📊 Analisis Sentimen")
    st.write("Upload dataset Anda dan lakukan analisis sentimen.")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_column = st.selectbox("Pilih kolom teks", df.columns)
        
        if st.button("Analisis Sentimen"):
            df['sentiment'] = df[text_column].astype(str).apply(analyze_sentiment)
            
            st.write("### Hasil Analisis Sentimen")
            st.dataframe(df[[text_column, 'sentiment']])
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil", csv, "hasil_analisis_sentimen.csv", "text/csv")
