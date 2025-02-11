import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def show():
    st.title("🧪 Uji Model Analisis Sentimen")
    st.write("Upload dataset, pilih model, dan lihat hasil evaluasi.")

    # Upload file
    uploaded_file = st.file_uploader("📂 Upload dataset CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Pastikan dataset memiliki kolom yang sesuai
        if 'clean_text' not in df.columns or 'sentiment' not in df.columns:
            st.error("Dataset harus memiliki kolom 'clean_text' dan 'sentiment'.")
            return
        
        # Hapus nilai NaN
        df.dropna(subset=['clean_text', 'sentiment'], inplace=True)
        
        # Menampilkan preview dataset
        st.write("📋 **Preview Data:**")
        st.dataframe(df.head())

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["clean_text"])
        y = df["sentiment"]

        # Split Data
        test_size = st.slider("📝 Pilih proporsi data uji (%)", 10, 50, 20, 5) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model selection
        model_options = {
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Neural Network": MLPClassifier(max_iter=300)
        }

        selected_model = st.selectbox("🔍 Pilih Model", list(model_options.keys()))

        if st.button("🚀 Jalankan Model"):
            model = model_options[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi Model
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Akurasi Model ({selected_model}): {accuracy:.4f}")

            # Plot Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", linewidths=0.5, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("Actual Labels")
            ax.set_title(f"Confusion Matrix - {selected_model}")
            st.pyplot(fig)

            # Classification Report
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).T.drop(["support"], axis=1)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Classes")
            ax.set_title(f"Classification Report - {selected_model}")
            st.pyplot(fig)
