import streamlit as st
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing_text import preprocessing_text
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
import pickle
from pickle import load

# Import data file model
# pakaian_vectorizer = pickle.load(open("model-tfidf/tf-idf-pakaian.pkl","rb"))


# def load_model_pred(model_file):
#     loaded_model = pickle.load(open(os.path.join(model_file)))    
#     return loaded_model

# def load_feature(model_file):
#     loaded_feature = pickle.load(open(os.path.join(model_file)))    
#     return  loaded_feature 

# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

# streamlit tampilan
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
    # loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(load(open("model-tfidf\feature_tf-idf-pakaian.pkl"))))
    loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open(os.path.join("model-tfidf\idf-pakaian.pkl"),'rb'))))

    
    model_pakaian=st.selectbox(
        "Pilih model",
        ('KNN', 'Decision tree', 'Naives bayes'))
    
    if model_pakaian=="Decision tree":
        st.write('Anda memilih Dec tree')
        predictor = pickle.load(open("model-tree\content\model_tree\model_pakaian\model_dec_pakaian_10persen.pkl",'rb'))
        if predictor is None:
            st.write("gagal")
        else:   
            prediction = predictor.predict(loaded_vec.fit_transform([data_input]))
            st.write("siap untuk di prediksi")
            st.write(prediction)
        # final_result = get_key(prediction)
        # st.success("News Categorized as:: {}".format(final_result))
        
            
    if model_pakaian=="KNN":
            st.write('Anda memilih KNN')
            st.write('', opini_pakaian)
    if model_pakaian=="Naives bayes":
            st.write('Anda memilih Naives bayes')
            st.write('', opini_pakaian)
            
    