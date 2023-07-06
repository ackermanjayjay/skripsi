import streamlit as st
import pandas as pd
import numpy as np

tab1, tab2 = st.tabs(["Pakaian", "Elektronik"])

with tab2:
    st.header('Sentimen analisis produk tokopedia')
    st.title('Produk Pakaian ')

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
    model_pakaian=st.selectbox(
        "Pilih model",
        ('KNN', 'Decision tree', 'Naives bayes'))
    
    if model_pakaian=="Decision tree":
            st.write('Anda memilih Dec tree')
            st.write('', opini_pakaian)
    if model_pakaian=="KNN":
            st.write('Anda memilih KNN')
            st.write('', opini_pakaian)
    if model_pakaian=="Naives bayes":
            st.write('Anda memilih Naives bayes')
            st.write('', opini_pakaian)
