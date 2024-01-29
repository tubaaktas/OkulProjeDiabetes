import streamlit as st
import plotly.express as px
import joblib
import pandas as pd

st.set_page_config(layout="wide",page_title="👨🏻‍⚕️ Diyabet Tahmini")


data = pd.read_csv('database/diabetes.csv')



@st.cache_data
def get_data():
    df = pd.read_csv('database/diabetes.csv')
    return df


def get_model():
    model = joblib.load("voting_model(lgbm_xgb_rf)_mean.pkl")
    return model


st.header("👨🏻‍⚕️ Diyabet Tahmini ")
tab_home, tab_info, tab_modelinfo,  tab_model = st.tabs(["Ana Sayfa", "Hakkımızda Bilgi", "Model Hakkında Bilgi", "Model"])

# TAB HOME
column_diabetes, column_dataset = tab_home.columns(2)

column_diabetes.subheader("Diyabet Nedir?")
column_diabetes.markdown(
    "Halk arasında genel olarak  şeker hastalığı olarak tabir edilen Diabetes Mellitus, genel olarak kanda glukoz (şeker) seviyesinin normalin üzerine çıkması,"
    " buna bağlı olarak normalde şeker içermemesi gereken idrarda şekere rastlanmasıdır. Farklı türevleri bulunan diyabet hastalığı, ülkemizde ve dünyada en sık"
    " rastlanan hastalıklar arasında yer alır. Uluslararası Diyabet Federasyonu'nun sağlamış olduğu istatistiki verilere göre her 11 yetişkinden biri diyabet "
    "hastalığına sahip olmakla birlikte her 6 saniyede 1 birey diyabet kaynaklı sorunlar nedeniyle hayatını kaybetmektedir.")
column_diabetes.subheader(" Kolonlar :")
column_diabetes.markdown("* Pregnancies = Hamile kalma sayısı (0-17)")
column_diabetes.markdown("* Glucose = Kan glukoz Değeri (0-199)")
column_diabetes.markdown("* Blood Pressure = Kan Basıncı (0-122 mm/Hg)")
column_diabetes.markdown("* Skin Thickness = Deri Kalınlığı (0-99 mm)")
column_diabetes.markdown("* Insulin = İnsulin Değeri (0-846 mu U/ml)")
column_diabetes.markdown("* BMI = Vücut Kitle İndeksi kg ve m^2")
column_diabetes.markdown(
    "* Diabetes Pedigree Function = Kişinin şeker hastalığına genetik olarak yatkınlığı (0.078-2.42)")
column_diabetes.markdown("* Age = Yas Değeri")
column_diabetes.markdown("* Outcome = Çıktı (0 / 1)")

column_dataset.subheader("Diabetes Veri Seti")
df = get_data()
column_dataset.dataframe(df)
# st.checkbox("Use container width", value=False, key="use_container_width")
# column_dataset.dataframe(get_data(), use_container_width = st.session_state.use_container_width)

# TAB INFO
column_Tugba, column_Temel = tab_info.columns(2)
column_Tugba.image('ara_yuz/galeri/tugba_pp.png', width=300)
column_Tugba.subheader("Tuğba AKTAŞ")
column_Tugba.markdown("Merhaba ben Tuğba Aktaş, Veri Bilimiyle ilgili araştırmalar yapıyor ve bu alanda mentor yardımcılığı yapıyorum. Yani yapıyorum herhalde.")
column_Tugba.markdown("LinkedIn : [TUAKTAS_LinkedIn](https://www.linkedin.com/in/tugbaaktas/)")
column_Tugba.markdown("Github : [TUAKTAS_Github](https://github.com/tubaaktas)", )


column_Temel.image('ara_yuz/galeri/temelinko.png', width=300)
column_Temel.subheader("İsmail Mert TEMEL")
column_Temel.markdown("Merhaba ben İsmail Mert TEMEL, Makine Öğrenmesi ve Veri Bilimi üzerine ufak çalışmalar yapmaya çalışıyorum. Mezun olup apartman yöneticisi olmak kariyerim adına en büyük planım.")
column_Temel.markdown("LinkedIn : [TEMELINKO_LinkedIn](https://www.linkedin.com/in/ismail-mert-temel-688abb198/)",unsafe_allow_html=True)
column_Temel.markdown("Github : [TEMELINKO_Github](https://github.com/Temelinko)")


# TAB MODEL İNFO

tab_modelinfo.subheader("Model İle İlgili Neler Yaptık")
tab_modelinfo.markdown("")

# TAB MODEL

model = get_model()

Insulin = tab_model.number_input("Insulin değeriniz", min_value=10, max_value=900)
Glucose = tab_model.number_input("Glucose değeriniz", min_value=10, max_value=250)
Age = tab_model.slider("Yaşınız", min_value=12, max_value=100)
BMI = tab_model.number_input("BMI değeriniz", min_value=1, max_value=100)
# NEW_INSULIN_Normal = tab_model.number_input("Insulin değeriniz normal mi ? (16-166 aralığındaysa 1 değilse 0)", min_value=0, max_value=1)

user_input = pd.DataFrame({'Insulin': Insulin, 'Glucose': Glucose, 'Age': Age, 'BMI': BMI}, index=[0])
# tab_model.write(user_input)


if tab_model.button("Tahmin Etmek İçin Basınız"):
    prediction = model.predict(user_input)
    # tab_model.success(f"tahmın edılen deger: {prediction[0]}")
    if prediction == 1:
        tab_model.image('ara_yuz/galeri/hemsire.png', width=600)
    else:
        tab_model.success("Diyabet Değilsiniz")
        tab_model.balloons()

