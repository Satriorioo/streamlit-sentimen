import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Fungsi untuk menampilkan visualisasi sentimen
def show():
    st.title("📊 Visualisasi Sentimen")
    st.write("Upload dataset hasil analisis sentimen untuk melihat visualisasi data.")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'sentiment' not in df.columns:
            st.error("Dataset harus memiliki kolom 'sentiment'.")
            return
        
        # Hitung jumlah sentimen
        sentiment_counts = df['sentiment'].value_counts()
        
        # Pie chart
        st.write("### Diagram Pie Sentimen")
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'gray'])
        st.pyplot(fig)
        
        # Bar chart
        st.write("### Diagram Batang Sentimen")
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['green', 'red', 'gray'], ax=ax)
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
        
        st.write("### Tabel Distribusi Sentimen")
        st.dataframe(sentiment_counts.to_frame().reset_index().rename(columns={'index': 'Sentimen', 'sentiment': 'Jumlah'}))

 # Word Cloud
        st.write("### Word Cloud")
        text_corpus = ' '.join(df['clean_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        # Frekuensi Kata
        st.write("### Diagram Frekuensi Kata")
        words = ' '.join(df['clean_text'].dropna()).split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)
        words, counts = zip(*common_words)
        fig, ax = plt.subplots()
        plt.bar(words, counts)
        plt.xticks(rotation=45)
        plt.ylabel("Frekuensi")
        st.pyplot(fig)
