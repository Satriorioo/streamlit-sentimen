import streamlit as st
import preprocessing
import sentiment_analysis
import visualization
import model_testing  

st.sidebar.title("📌 Navigasi Menu")
menu = st.sidebar.radio("Pilih Halaman", 
                        ["🏠 Dashboard", "📝 Preprocessing Teks", "📊 Analisis Sentimen", "📈 Visualisasi Data", "🧪 Uji Model"])

if menu == "🏠 Dashboard":
    st.title("📊 Aplikasi Analisis Sentimen")
    st.write("Selamat datang di **Aplikasi Analisis Sentimen** yang telah dibuat oleh Bowo. Aplikasi ini dirancang untuk membantu dalam menganalisis opini atau sentimen dari teks menggunakan metode pemrosesan bahasa alami (NLP).")
    
    st.subheader("📖 Apa itu Analisis Sentimen?")
    st.write("""
    Analisis sentimen adalah proses memahami, mengekstraksi, dan mengkategorikan opini dari teks berdasarkan polaritasnya, seperti positif, negatif, atau netral. 
    Aplikasi ini menggunakan teknik preprocessing teks dan metode lexicon-based untuk menentukan sentimen dari data yang diberikan.
    """)

    st.subheader("🔹 Fitur yang tersedia:")
    st.markdown("""
    - **📝 Preprocessing Teks**: Membersihkan data teks dari URL, HTML, emoji, tanda baca, angka, serta melakukan stemming dan normalisasi.
    - **📊 Analisis Sentimen**: Menentukan apakah teks memiliki sentimen positif, negatif, atau netral berdasarkan metode lexicon-based.
    - **📈 Visualisasi Data**: Menampilkan hasil analisis dalam bentuk grafik pie, bar chart, word cloud, dan frekuensi kata.
    - **🧪 Uji Model**: Menguji model analisis sentimen pada teks inputan secara real-time.
    """)

elif menu == "📝 Preprocessing Teks":
    preprocessing.show()

elif menu == "📊 Analisis Sentimen":
    sentiment_analysis.show()

elif menu == "📈 Visualisasi Data":
    visualization.show()

elif menu == "🧪 Uji Model":
    model_testing.show()  
