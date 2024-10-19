# Nama 	: Ataka Dzulfikar
# NIM  	: 22537141002
# Prodi	: Teknologi Informasi / I

# Impport library yang diperlukan
import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Fungsi untuk mengubah teks menjadi vektor fitur
# Fungsi untuk analisis sentimen menggunakan Naive Bayes
def analisis_sentimen_nb(text, model, vectorizer):
    # Transformasi teks input menjadi vektor
    X = vectorizer.transform([text])
    # Prediksi sentimen dari model Naive Bayes
    result = model.predict(X)[0]

    # Menentukan hasil sentimen berdasarkan prediksi
    if result == 1:
        return "Positif"
    elif result == 0:
        return "Netral"
    elif result == -1:
        return "Negatif"
    else:
        return "Tidak Diketahui" 

def main():
    st.title("Analisis Sentimen menggunakan Streamlit")
    # merupakan side bar yang digunakan
    menu = ["Beranda", "Tentang"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Beranda":
        st.subheader("Beranda")

        # Try loading the dataset
        try:
            data = pd.read_csv("sentiment_twitter.csv", sep='\t', on_bad_lines='skip')  # Gunakan tab sebagai pemisah
            st.write("Data loaded successfully!")
            st.write("Kolom yang tersedia: ", data.columns.tolist())  # Tampilkan kolom
            data.columns = data.columns.str.strip()  # Hapus spasi di sekitar nama kolom jika ada
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        # Pastikan kolom yang benar digunakan
        if 'sentimen' in data.columns and 'Tweet' in data.columns:
            tweet_column = 'Tweet'  # Gunakan kolom 'Tweet' untuk analisis
        else:
            st.error("Kolom 'sentimen' atau 'Tweet' tidak ditemukan. Pastikan nama kolom sesuai.")
            return

        # Preprocessing untuk Naive Bayes
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data[tweet_column])  # Proses teks dari kolom 'Tweet'
        model = MultinomialNB()

        # Melatih model Naive Bayes
        try:
            model.fit(X, data['sentimen'])  # Gunakan kolom 'sentimen' sebagai label
        except Exception as e:
            st.error(f"Error during model training: {e}")
            return

        # Form untuk analisis sentimen
        with st.form("nlpForm"):
            raw_text = st.text_area("Masukkan kalimat Anda")  # Input teks untuk analisis
            submit_button = st.form_submit_button(label='Cek Disini')

        # Apabila form dikirimkan
        if submit_button and raw_text:
            col1 = st.columns(1)[0]

            with col1:
                st.info("Naive Bayes Sentiment Analysis")
                sentiment_nb = analisis_sentimen_nb(raw_text, model, vectorizer)
                st.write(f"Sentimen Naive Bayes dari teks: {sentiment_nb}")

    elif choice == "Tentang":
        st.subheader("Tentang")
        st.write("Ini adalah aplikasi analisis sentimen yang menggunakan Naive Bayes dan TextBlob.")

if __name__ == '__main__':
    main()
