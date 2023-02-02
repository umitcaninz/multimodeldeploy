import pickle
import streamlit as st
from streamlit_option_menu import option_menu

diabetes_model = pickle.load(open("diabetes_model.sav","rb"))

stroke_model = pickle.load(open("stroke_model.sav","rb"))


with st.sidebar:

    selected = option_menu("Makine Öğrenmesi Kullanarak Hastalık Tahmin Etme",
                           ["UMIT CAN INOZU","Diyabet Tahmini",
                            ],
                           icons= ["person","activity"],
                           default_index=0)


#Diyabet tahmin sayfası

if (selected == "UMIT CAN INOZU"):
    with st.container():
        st.subheader("Merhabalar , Ben Ümit Can İNÖZÜ")
        st.markdown("Ankara Üniversitesi İstatistik bölümü 3.sınıf öğrencisiyim. Okulda öğrendiğim teorik istatistik bilgisi ve kendi başıma öğrendiğim yazılım bilgimi birleştirerek kendimi Veri Bilimi alanında projeler yaparak geliştiriyorum.")
        st.markdown("[LınkedIn >](linkedin.com/in/ümit-can-inözü/)")
        st.markdown("[Kaggle >]( kaggle.com/umitcaninz)")
        st.markdown("[GıtHub >](github.com/umitcaninz/)")






if (selected == "Diyabet Tahmini"):

    st.title("Makine Öğrenmesi Diyabet Tahmini")

    col1 , col2 , col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Hamilelik Sayısı")
    with col2:
        Glucose = st.text_input("Glucose Değeri")
    with col3:
        BloodPressure = st.text_input("Kan Basıncı Değeri")
    with col1:
        SkinThickness = st.text_input("Cilt Kalınlığı Değeri")
    with col2:
        Insulin = st.text_input("Insulin Değeri")
    with col3:
        BMI = st.text_input("BMI Değeri")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Değeri")
    with col2:
        Age = st.text_input("Yaş")


    diab_dignosis = ""


    if st.button("Diyabet Test Sonuçları"):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diab_prediction[0]==1):
            diab_dignosis = "Diyabet Hastasısınız"

        else:
            diab_dignosis = "Diyabet Hastası değilsiniz"


    st.success(diab_dignosis)














