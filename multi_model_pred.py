import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import requests
from PIL import Image

st.set_page_config(page_title="diabetes")

diabetes_model = pickle.load(open("diabetes_model.sav","rb"))

stroke_model = pickle.load(open("stroke_model.sav","rb"))


with st.sidebar:

    selected = option_menu("MACHINE LEARNING MODEL DEPLOYMENT",
                           ["ÜMİT CAN İNÖZÜ","DIABETES PREDICTION"
                            ],
                           icons= ["person","activity"],
                           default_index=0)





#Diyabet tahmin sayfası

if (selected == "ÜMİT CAN İNÖZÜ"):
    with st.container():

        col1,col2,col3 = st.columns(3)

        with col2 :
            img_contact_form = st.image("images/umitcaninz.png")


        st.subheader("Hi , I am Ümit Can İNÖZÜ :wave:")
        st.markdown("I am a 3rd year student at Ankara University, Department of Statistics. I develop myself by doing projects in the field of Data Science by combining the theoretical statistics knowledge I learned at school and the software knowledge I learned on my own.")

        col1 , col2 , col3 = st.columns(3)

        with col1:
            st.markdown(
                "[![LinkedIn>](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/ümit-can-inözü/)")

        with col2:
            st.markdown("[![Kaggle >](https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-47.png)](https://www.kaggle.com/umitcaninz)")
        with col3:
            st.markdown("[![GitHub >](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/umitcaninz)")
##



    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html = True)

    local_css("style/style.css")

    with st.container():
        st.subheader("Connect with me: :")

        contact_form = """
        <form action="https://formsubmit.co/umitcaninozu@gmail.com" method="POST">
            <input type="text" name="name" placeholder = "Name" required>
            <input type="email" name="email" placeholder = "Email" required>
            <textarea name="message" placeholder = "Text Message" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        left_column , right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form,unsafe_allow_html=True)
        with right_column:
            st.empty()








if (selected == "DIABETES PREDICTION"):

    st.title(" Diabetes Prediction with Machine Learning")

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


    if st.button("Diabetes Prediction Results"):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diab_prediction[0]==1):
            diab_dignosis = "Diyabet Hastasısınız"

        else:
            diab_dignosis = "Diyabet Hastası değilsiniz"


    st.success(diab_dignosis)

