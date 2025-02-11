import streamlit as st
import pandas as pd
import re
import string
import emoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('stopwords')

# Load Stopwords dari file yang ada di folder program
STOPWORDS_FILE = "stopwords.txt"

def load_stopwords(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        st.error(f"⚠️ File {file_path} tidak ditemukan! Pastikan file ada di folder program.")
        return set()

stop_words = load_stopwords(STOPWORDS_FILE)

# Dictionary Normalisasi
normalization_dict = {
    "yg": "yang", "dg": "dengan", "tdk": "tidak", "ga": "tidak", "gk": "tidak","jg": "juga",
    "klo": "kalau", "org": "orang", "pd": "pada", "skrg": "sekarang", "sy": "saya",
    "utk": "untuk", "bgt": "banget", "dpt": "dapat", "jd": "jadi", "km": "kamu",
    "krn": "karena","kpn": "kapan", "lg": "lagi", "mksd": "maksud", "nggak": "tidak", "sblm": "sebelum",
    "sdh": "sudah", "spt": "seperti", "tp": "tapi", "udh": "sudah", "bs": "bisa",
    "bnyk": "banyak", "btw": "by the way", "cpt": "cepat", "dr": "dari", "emg": "memang",
    "gmn": "gimana", "gt": "gitu", "gw": "saya", "hrs": "harus", "kek": "seperti",
    "kl": "kalau", "knp": "kenapa", "ky": "kayak", "lbh": "lebih", "lw": "lu",
    "mksh": "terima kasih", "mn": "mana", "mngkn": "mungkin", "msh": "masih",
    "ntar": "nanti", "nya": "nya", "pdhl": "padahal", "plg": "paling", "prnh": "pernah",
    "trs": "terus", "ttg": "tentang", "wkt": "waktu", "zmn": "zaman"
}

# Fungsi preprocessing
def remove_url(text):
    return re.sub(r'http[s]?://\S+', '', text)

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_number(text):
    return re.sub(r'\d+', '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)

def case_folding(text):
    return text.lower()

def normalize_text(text, normalization_dict):
    for word, replacement in normalization_dict.items():
        text = re.sub(r'\b' + word + r'\b', replacement, text)
    return text

def remove_stopwords(text, stop_words):
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(text):
    return stemmer.stem(text)

# Fungsi utama untuk ditampilkan di dashboard
def show():
    st.title("📝 Preprocessing Teks")
    st.write("Upload dataset Anda dan lakukan preprocessing teks.")

    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_column = st.selectbox("📝 Pilih kolom teks", df.columns)

        if st.button("🚀 Proses Preprocessing"):
            df['clean_text'] = df[text_column].astype(str)
            df['clean_text'] = df['clean_text'].apply(remove_url)
            df['clean_text'] = df['clean_text'].apply(remove_html)
            df['clean_text'] = df['clean_text'].apply(remove_emoji)
            df['clean_text'] = df['clean_text'].apply(remove_punctuation)
            df['clean_text'] = df['clean_text'].apply(remove_number)
            df['clean_text'] = df['clean_text'].apply(remove_mentions)  
            df['clean_text'] = df['clean_text'].apply(remove_hashtags)  
            df['clean_text'] = df['clean_text'].apply(case_folding)
            df['clean_text'] = df['clean_text'].apply(lambda x: normalize_text(x, normalization_dict))
            df['clean_text'] = df['clean_text'].apply(lambda x: remove_stopwords(x, stop_words))
            df['clean_text'] = df['clean_text'].apply(stemming)

            st.write("### ✅ Hasil Preprocessing")
            st.dataframe(df[['clean_text']])

            # Word Cloud
            st.write("### ☁️ Word Cloud")
            text_corpus = ' '.join(df['clean_text'].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            # Frekuensi Kata
            st.write("### 📊 Diagram Frekuensi Kata")
            words = ' '.join(df['clean_text'].dropna()).split()
            word_counts = Counter(words)
            common_words = word_counts.most_common(10)
            words, counts = zip(*common_words)
            fig, ax = plt.subplots()
            plt.bar(words, counts)
            plt.xticks(rotation=45)
            plt.ylabel("Frekuensi")
            st.pyplot(fig)

            # Download hasil preprocessing
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil", csv, "hasil_preprocessing.csv", "text/csv")
