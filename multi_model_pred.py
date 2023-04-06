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
                           ["ÜMİT CAN İNÖZÜ","DIABETES PREDICTION","TITANIC PREDICTON"
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


if(selected=="TITANIC PREDICTION"):
    import streamlit as st
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_validate, RandomizedSearchCV, validation_curve

    data = pd.read_csv("titanic.csv")
    data.head()
    data = data.dropna()
    df = data.copy()

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(df[features])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)
    lgbm_model = LGBMClassifier(random_state=17)

    lgbm_final = lgbm_model.set_params(learning_rate=0.01, colsample_bytree=0.7, n_estimators=300, random_state=17).fit(
        X, y)
    cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    accuracy = cv_results["test_accuracy"].mean()
    predict = lgbm_final.predict(X_test)
    import streamlit as st
    from PIL import Image

    # Arka plan rengi ve yazı fontunu değiştirmek
    st.set_page_config(page_title="Titanic", page_icon=":ship:", layout="wide", initial_sidebar_state="expanded")
    st.title("Titanic Prediction with Machine Learning")
    st.markdown("<h3 style='text-align: center; color: #908f8f;'>Accuracy Score: {:.2f}</h3>".format(accuracy),
                unsafe_allow_html=True)

    # Resim ekleme
    image = Image.open("titanic.jpg")
    st.image(image, use_column_width=True)

    # Tahmin yapılacak özelliklerin girilmesi
    st.sidebar.markdown("<h4 style='color: #908f8f;,background-color: #000000;'>Prediction</h4>",
                        unsafe_allow_html=True)

    sex = st.sidebar.selectbox("Cinsiyet", ["Erkek", "Kadın"])
    pclass = st.sidebar.selectbox("Seyahat Sınıfı", [1, 2, 3])
    siblings_spouses = st.sidebar.slider("Kardeş / Eş Sayısı", 0, 8, 0)
    parents_children = st.sidebar.slider("Ebeveyn / Çocuk Sayısı", 0, 6, 0)

    # Düğme ekleyerek tahminin gösterilmesi
    if st.sidebar.button("Tahmin Et"):
        input = pd.DataFrame(
            {"Pclass": [pclass], "Sex": [sex], "SibSp": [siblings_spouses], "Parch": [parents_children]})
        input = pd.get_dummies(input)
        input = input.reindex(columns=X.columns, fill_value=0)
        prediction = lgbm_final.predict(input)
        st.markdown("<h2 style='text-align: center; color: #908f8f;'>Prediction Result: {}</h2>".format(
            "Hayatta kalır" if prediction[0] == 1 else "Hayatta kalamaz"), unsafe_allow_html=True)












