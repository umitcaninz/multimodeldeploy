import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
from PIL import Image



diabetes_model = pickle.load(open("diabetes_model.sav","rb"))

stroke_model = pickle.load(open("stroke_model.sav","rb"))


with st.sidebar:

    selected = option_menu("Makine Öğrenmesi Model Deployment",
                           ["UMIT CAN INOZU","Diyabet Tahmini",
                            ],
                           icons= ["person","activity"],
                           default_index=0)





#Diyabet tahmin sayfası

if (selected == "UMIT CAN INOZU"):
    with st.container():

        col1,col2,col3 = st.columns(3)

        with col2 :
            img_contact_form = st.image("images/umitcaninz.png")


        st.subheader("Merhabalar , Ben Ümit Can İNÖZÜ :wave:")
        st.markdown("Ankara Üniversitesi İstatistik bölümü 3.sınıf öğrencisiyim. Okulda öğrendiğim teorik istatistik bilgisi ve kendi başıma öğrendiğim yazılım bilgimi birleştirerek kendimi Veri Bilimi alanında projeler yaparak geliştiriyorum.")

        col1 , col2 , col3 = st.columns(3)

        with col1:
            st.markdown(
                "[![LinkedIn>](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/ümit-can-inözü/)")

        with col2:
            st.markdown("[![Kaggle >](https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-47.png)](https://www.kaggle.com/umitcaninz)")
        with col3:
            st.markdown("[![GitHub >](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/umitcaninz)")




    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html = True)

    local_css("style/style.css")

    with st.container():
        st.header("Bana Ulaşmak İçin :")

        contact_form = """
        <form action="https://formsubmit.co/umitcaninozu@gmail.com" method="POST">
            <input type="text" name="name" placeholder = "İsminizi giriniz." required>
            <input type="email" name="email" placeholder = "Emailinizi giriniz." required>
            <textarea name="message" placeholder = "Mesajınızı yazınız." required></textarea>
            <button type="submit">Gönder</button>
        </form>
        """
        left_column , right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form,unsafe_allow_html=True)
        with right_column:
            st.empty()








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














