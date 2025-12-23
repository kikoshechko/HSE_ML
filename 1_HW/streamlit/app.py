import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("🚗 Прогноз цены автомобиля")
st.write("EDA + Ridge Regression модель")


# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load("best_model_pipeline.pkl")


model = load_model()

# Sidebar
section = st.sidebar.radio("Навигация", ["EDA", "Предсказание", "Веса модели"])

# EDA
if section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Первые строки датасета")
        st.dataframe(df.head())

        num_features = [
            "year",
            "km_driven",
            "mileage",
            "engine",
            "max_power",
            "selling_price",
        ]

        st.subheader("Гистограммы числовых признаков")

        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()

        for ax, col in zip(axes, num_features):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)

        st.pyplot(fig)

    else:
        st.info("Загрузите CSV для отображения EDA")

# Предсказание
elif section == "Предсказание":
    st.header("🔮 Предсказание цены")

    mode = st.radio("Способ ввода данных:", ["Ручной ввод", "Загрузка CSV"])

    cat_features = ["fuel", "seller_type", "transmission", "owner", "seats"]

    num_features = ["year", "km_driven", "mileage", "engine", "max_power"]

    all_features = num_features + cat_features

    if mode == "Ручной ввод":
        st.subheader("Введите признаки автомобиля")

        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Год выпуска", 1990, 2024, 2018)
            km_driven = st.number_input("Пробег", 0, 500000, 50000)
            mileage = st.number_input("Расход топлива", 0.0, 40.0, 18.0)
            engine = st.number_input("Объем двигателя", 500, 5000, 1200)
            max_power = st.number_input("Мощность", 30.0, 500.0, 80.0)

        with col2:
            fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "CNG", "LPG"])
            seller_type = st.selectbox(
                "Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"]
            )
            transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
            owner = st.selectbox(
                "Владелец", ["First Owner", "Second Owner", "Third Owner"]
            )
            seats = st.selectbox("Кол-во мест", [4, 5, 7])

        input_df = pd.DataFrame(
            [
                {
                    "year": year,
                    "km_driven": km_driven,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": max_power,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                    "seats": seats,
                }
            ]
        )

        if st.button("Предсказать цену"):
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Предсказанная цена: **{prediction:,.0f}**")

    else:
        st.subheader("Загрузите CSV с признаками")

        uploaded_file = st.file_uploader("CSV без target и name", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            preds = model.predict(df)
            df["predicted_price"] = preds

            st.subheader("Результаты")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать результат", csv, "predictions.csv", "text/csv")

# Веса модели
elif section == "Веса модели":
    st.header("⚖️ Веса Ridge Regression")

    preprocessor = model.named_steps["preprocessor"]
    ridge = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = ridge.coef_

    coef_df = pd.DataFrame({"feature": feature_names, "weight": coefs}).sort_values(
        "weight", key=abs, ascending=False
    )

    st.subheader("Топ-20 признаков по |весу|")
    st.dataframe(coef_df.head(20))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=coef_df.head(20), x="weight", y="feature", ax=ax)
    ax.set_title("Коэффициенты модели Ridge")

    st.pyplot(fig)
