import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Заголовок приложения
st.title("Прогнозирование цены автомобиля")
st.markdown("Приложение использует обученную Ridge-модель с предобработкой признаков (OHE + StandardScaler)")

# Sidebar - выбор источника данных
st.sidebar.header("Настройки приложения")

input_method = st.sidebar.radio(
    "Выберите способ ввода данных:",
    ("Загрузить CSV", "Ручной ввод")
)
# Загрузка модели
@st.cache_data
def load_model(pickle_path):
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)
    return model

model_path = Path.home() / "Desktop/Машинное обучение/1 HW/models/best_model_pipeline.pkl"
model_pipeline = load_model(model_path)
st.sidebar.success("Модель загружена")
# Загрузка данных или ручной ввод
if input_method == "Загрузить CSV":
    uploaded_file = st.file_uploader("Загрузите CSV с признаками автомобиля", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Превью загруженных данных")
        st.dataframe(df.head())

elif input_method == "Ручной ввод":
    st.sidebar.subheader("Введите характеристики автомобиля")
    year = st.sidebar.number_input("Год выпуска", 1980, 2025, 2015)
    km_driven = st.sidebar.number_input("Пробег (км)", 0, 500000, 50000)
    mileage = st.sidebar.number_input("Расход (км/л)", 0.0, 50.0, 18.0)
    engine = st.sidebar.number_input("Объем двигателя (см³)", 500, 4000, 1200)
    max_power = st.sidebar.number_input("Мощность двигателя (л.с.)", 10, 400, 75)
    fuel = st.sidebar.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG"])
    seller_type = st.sidebar.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.sidebar.selectbox("Коробка передач", ["Manual", "Automatic"])
    owner = st.sidebar.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    seats = st.sidebar.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10, 14])

    # Формируем DataFrame для модели
    df = pd.DataFrame([{
        'year': year,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'seats': seats
    }])
    st.subheader("Введенные данные")
    st.dataframe(df)
# Кнопка предсказания
if df is not None:
    if st.button("Сделать прогноз"):
        predictions = model_pipeline.predict(df)
        df['predicted_price'] = predictions
        st.subheader("Предсказанная цена")
        st.write(df[['predicted_price']])
#  Визуализация коэффициентов
if st.checkbox("Показать веса/коэффициенты модели"):
    coefs = model_pipeline.named_steps['model'].coef_
    cat_features = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    num_features = model_pipeline.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_
    feature_names = list(num_features) + list(cat_features)

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs
    }).sort_values(by='coefficient', ascending=False)

    st.subheader("Коэффициенты признаков")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='coefficient', y='feature', data=coef_df, ax=ax)
    st.pyplot(fig)
    
# EDA-графики (опционально)
if st.checkbox("Показать ключевые графики EDA"):
    st.subheader("Распределение числовых признаков")
    numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8, len(numeric_cols)*3))
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Распределение {col}")
    plt.tight_layout()
    st.pyplot(fig)
