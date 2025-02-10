import streamlit as st
import pandas as pd

# Fungsi untuk menampilkan halaman download hasil

def show():
    st.title("📥 Download Hasil Analisis Sentimen")
    st.write("Upload dataset hasil analisis sentimen untuk mengunduh hasilnya.")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'sentiment' not in df.columns:
            st.error("Dataset harus memiliki kolom 'sentiment'.")
            return
        
        st.write("### Pratinjau Hasil")
        st.dataframe(df.head())
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil", csv, "hasil_analisis_sentimen.csv", "text/csv")
