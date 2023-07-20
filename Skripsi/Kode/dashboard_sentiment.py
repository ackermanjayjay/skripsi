import streamlit as st
import pandas as pd
from preprocessing_text import preprocessing_text
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from pickle import load
import pickle

tab1, tab2 = st.tabs(["Pakaian", "Elektronik"])

with tab2:
    st.header('Sentimen analisis produk tokopedia')
    st.title('Produk Elektronik')

    opini_elektornik = st.text_input('Masukkan opini ini untuk Produk Elektronik')
    model_elektronik=st.radio(
        "Pilih model",
        ('KNN', 'Decision tree', 'Naives bayes'))
    if model_elektronik=="Decision tree":
            st.write('Anda memilih Dec tree')
            st.write('',  opini_elektornik)
    if model_elektronik=="KNN":
            st.write('Anda memilih KNN')
            st.write('', opini_elektornik)
    if model_elektronik=="Naives bayes":
            st.write('Anda memilih Naives bayes')
            st.write('',  opini_elektornik)
    

with tab1:
    st.header('Sentimen analisis produk tokopedia')
    st.title('Produk Pakaian ')

    opini_pakaian = st.text_input('Masukkan opini ini untuk Produk Pakaian')
    data_input = preprocessing_text(opini_pakaian)
    tfidf = TfidfVectorizer

    
    model_pakaian=st.selectbox(
        "Pilih model",
        ('KNN', 'Decision tree', 'Naives bayes'))
    
    if model_pakaian=="Decision tree":
            st.write('Anda memilih Dec tree')
            loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("model-tfidf\feature_tf-idf-pakaian.pkl", "rb"))))
            
            pipeline_dec_tree = load("model-tree\model_dec_pakaian_40persen.joblib")

            hasil = pipeline_dec_tree.predict(loaded_vec.fit_transform([data_input]))
            st.write('', opini_pakaian)
            st.write('', hasil)
            
    if model_pakaian=="KNN":
            st.write('Anda memilih KNN')
            st.write('', opini_pakaian)
    if model_pakaian=="Naives bayes":
            st.write('Anda memilih Naives bayes')
            st.write('', opini_pakaian)
