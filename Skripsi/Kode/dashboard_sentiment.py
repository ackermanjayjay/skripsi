import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing_text import preprocessing_text
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
import pickle
from pickle import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import pandas as pd
from io import StringIO


# streamlit tampilan
tab1, tab2 = st.tabs(["Pakaian", "Elektronik"])

# Tab 2 untuk prediksi data elektronik
with tab2:
    st.header('Sentimen analisis produk tokopedia')
    st.title('Produk Elektronik')

    opini_elektronik = st.text_input(
        'Masukkan opini ini untuk Produk Elektronik')

    opini_elektronik_preprocess = preprocessing_text(opini_elektronik)

    loaded_vec_elektronik = TfidfVectorizer(decode_error="replace", vocabulary=set(
        pickle.load(open(os.path.join("model-tfidf\idf-elektronik.pkl"), 'rb'))))

    tfidf_elektronik = loaded_vec_elektronik.fit_transform(
                [opini_elektronik_preprocess])
     
    model_elektronik = st.radio(
        "Pilih model",
        ('KNN', 'Decision tree', 'Naives bayes'))

    if model_elektronik == "Decision tree":
        st.write('Anda memilih Dec tree')
        predictor_elektronik_tree = pickle.load(open(
            "model-tree\content\model_tree\model_elektronik\model_dec_elektronik_40persen.pkl", 'rb'))
        if predictor_elektronik_tree is None:
            st.write("gagal")
        else:
            st.write("siap untuk di prediksi")
           

            prediction_elektronik = predictor_elektronik_tree.predict(
                tfidf_elektronik)
            st.dataframe({"komentar": opini_elektronik,
                          "prediksi": prediction_elektronik})

    if model_elektronik == "KNN":
        st.write('Anda memilih KNN')
        predictor_elektronik_knn = pickle.load(open(
            "model-knn\content\model_KNN\k_4\elektronik\model_knn_elektronik_10persen_data_K4.pkl", 'rb'))
        if predictor_elektronik_knn is None:
            st.write("gagal")
        else:
            tfidf_elektronik = loaded_vec_elektronik.fit_transform(
                [opini_elektronik])

            prediction_elektronik_knn = predictor_elektronik_knn.predict(
                tfidf_elektronik)
            st.dataframe({"komentar": opini_elektronik,
                          "prediksi": prediction_elektronik_knn})

    if model_elektronik == "Naives bayes":
        st.write('Anda memilih Naives bayes')
        st.write('',  opini_elektronik)


# Tab 1 untuk data pakaian
with tab1:
    st.header('Sentimen analisis produk tokopedia')
    st.title('Produk Pakaian ')

    opini_pakaian = st.text_input('Masukkan opini ini untuk Produk Pakaian')

    opini_pakaian_preprocess = preprocessing_text(opini_pakaian)
    loaded_vec_pakaian = TfidfVectorizer(decode_error="replace", vocabulary=set(
        pickle.load(open(os.path.join("model-tfidf\idf-pakaian.pkl"), 'rb'))))
    tfidf_pakaian = loaded_vec_pakaian.fit_transform(
        [opini_pakaian_preprocess])
    
    model_pakaian = st.selectbox(
        "Pilih model",
        ('KNN', 'Decision tree', 'Naives bayes'))

    if model_pakaian == "Decision tree":
        st.write('Anda memilih Dec tree')
        predictor_load_dec_pakaian = pickle.load(open(
            "model-tree\content\model_tree\model_pakaian\model_dec_pakaian_10persen.pkl", 'rb'))
        if predictor_load_dec_pakaian is None:
            st.write("gagal")
        else:
            prediction_dec_pakaian = predictor_load_dec_pakaian.predict(
                tfidf_pakaian)
            st.write("siap untuk di prediksi")
            st.dataframe({"opini anda": opini_pakaian,
                          "prediksi":  prediction_dec_pakaian})

    if model_pakaian == "KNN":
        st.write('Anda memilih KNN')
        predictor_load_pakaian_knn = pickle.load(open(
            "model-knn\content\model_KNN\k_4\pakaian\model_knn_pakaian_10persen_data_K4.pkl", 'rb'))
        if predictor_load_pakaian_knn is None:
            st.write("gagal")
        else:
            prediction_pakaian_knn = predictor_load_pakaian_knn.predict(
                tfidf_pakaian)
            st.dataframe({"komentar": opini_pakaian,
                          "prediksi": prediction_pakaian_knn})
    if model_pakaian == "Naives bayes":
        st.write('Anda memilih Naives bayes')
        st.write('', opini_pakaian)
