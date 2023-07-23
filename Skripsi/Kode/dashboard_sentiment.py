import streamlit as st
# import pandas as pd
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

# Tab elektronik
with tab2:
    st.header('Sentimen analisis produk tokopedia')
    st.title('Produk Elektronik')

    # opini_elektornik = st.text_input(
    #     'Masukkan opini ini untuk Produk Elektronik')
    upload_file_data_elektronik = st.file_uploader("choose a file")
    if upload_file_data_elektronik is not None:
        dataframe_elektronik = pd.read_csv(upload_file_data_elektronik)
        st.write( dataframe_elektronik["komentar"])

    # untuk memasukkan lewat form
    # data_input_elektronik = preprocessing_text(opini_elektornik)
    # untuk upload data berbentuk csv
    dataframe_elektronik["komentar_bersih"] =  dataframe_elektronik["komentar"].apply(preprocessing_text)
    st.write(dataframe_elektronik["komentar_bersih"] )
    loaded_vec_elektronik = TfidfVectorizer(decode_error="replace", vocabulary=set(
        pickle.load(open(os.path.join("model-tfidf\idf-elektronik.pkl"), 'rb'))))

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
            # untuk tfidf data opini dari form
            # prediction_elektronik_tree_40persen = predictor_elektronik_tree.predict(
            #     loaded_vec_elektronik.fit_transform([data_input_elektronik]))
            # st.write(classification_report(
            #     prediction_elektronik_tree_40persen, prediction_elektronik_tree_40persen))
            # st.write(prediction_elektronik_tree_40persen)
            
            # untuk tfidf data opini dari dataframe csv
            df_tfidf=dataframe_elektronik["komentar_bersih"]
            tfidf_elektronik_df=loaded_vec_elektronik.fit_transform(df_tfidf)
            prediction_elektronik_df = predictor_elektronik_tree.predict(tfidf_elektronik_df)
            st.dataframe({"komentar":df_tfidf,
                          "prediksi":prediction_elektronik_df})

    # if model_elektronik == "KNN":
    #     st.write('Anda memilih KNN')
    #     st.write('', opini_elektornik)
    # if model_elektronik == "Naives bayes":
    #     st.write('Anda memilih Naives bayes')
    #     st.write('',  opini_elektornik)


with tab1:
    st.header('Sentimen analisis produk tokopedia')
    st.title('Produk Pakaian ')

    opini_pakaian = st.text_input('Masukkan opini ini untuk Produk Pakaian')

    data_input = preprocessing_text(opini_pakaian)
    loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(
        pickle.load(open(os.path.join("model-tfidf\idf-pakaian.pkl"), 'rb'))))

    model_pakaian = st.selectbox(
        "Pilih model",
        ('KNN', 'Decision tree', 'Naives bayes'))

    if model_pakaian == "Decision tree":
        st.write('Anda memilih Dec tree')
        predictor = pickle.load(open(
            "model-tree\content\model_tree\model_pakaian\model_dec_pakaian_10persen.pkl", 'rb'))
        if predictor is None:
            st.write("gagal")
        else:
            tfidf_pakaian = loaded_vec.fit_transform([data_input])
            prediction = predictor.predict(tfidf_pakaian)
            st.write("siap untuk di prediksi")
            st.write(prediction)
            # st.write(classification_report(loaded_vec.fit_transform([data_input]),prediction))

    if model_pakaian == "KNN":
        st.write('Anda memilih KNN')
        st.write('', opini_pakaian)
    if model_pakaian == "Naives bayes":
        st.write('Anda memilih Naives bayes')
        st.write('', opini_pakaian)
