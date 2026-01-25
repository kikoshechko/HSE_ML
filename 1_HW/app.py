import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error as MSE

st.set_page_config(page_title="Car Price Prediction", layout="wide")

# --- Функции ---
def plot_eda(df):
    st.subheader("Основные графики EDA")
    num_features = df.select_dtypes(include=np.number).columns.tolist()
    cat_features = df.select_dtypes(exclude=np.number).columns.tolist()

    if num_features:
        with st.expander("Числовые признаки"):
            fig, axes = plt.subplots(len(num_features), 1, figsize=(6, 4*len(num_features)))
            if len(num_features) == 1:
                axes = [axes]
            for i, col in enumerate(num_features):
                sns.histplot(df[col].dropna(), ax=axes[i], kde=True)
                axes[i].set_title(col)
            st.pyplot(fig)
    else:
        st.info("В данных нет числовых признаков для отображения.")

    if cat_features:
        with st.expander("Категориальные признаки"):
            for col in cat_features:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x=col, order=df[col].value_counts().index)
                ax.set_title(col)
                st.pyplot(fig)
    else:
        st.info("В данных нет категориальных признаков для отображения.")

def train_model(df):
    target = 'selling_price'

    cat_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    num_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

    X = df.drop(columns=['selling_price', 'name'], errors='ignore')
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ],
        remainder='drop'
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {'model__alpha': np.logspace(-4, 2, 10)}

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_

def predict_single(model, input_df):
    return model.predict(input_df)

# --- Заголовок ---
st.title("Предсказание цены автомобиля")

# --- Шаг 1: загрузка обучающего CSV ---
st.header("Шаг 1: Загрузите обучающий CSV")
train_file = st.file_uploader("Файл CSV с обучающими данными", type="csv")

if train_file:
    df_train = pd.read_csv(train_file)
    st.success("Файл загружен!")
    plot_eda(df_train)

    with st.expander("Обучение модели"):
        best_model, best_params = train_model(df_train)
        st.write(f"Лучшие параметры модели: {best_params}")

        # Визуализация коэффициентов Ridge
        st.subheader("Веса модели (коэффициенты)")
        try:
            coefs = best_model.named_steps['model'].coef_
            # Получаем имена признаков после OneHotEncoder
            ohe_features = best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
            all_features = best_model.named_steps['preprocessor'].transformers_[0][2] + list(ohe_features)
            coef_df = pd.DataFrame({'feature': all_features, 'coef': coefs})
            coef_df = coef_df.sort_values(by='coef', key=abs, ascending=False)
            st.bar_chart(coef_df.set_index('feature')['coef'])
        except Exception as e:
            st.warning("Не удалось отобразить коэффициенты модели.")

# --- Шаг 2: предсказание ---
st.header("Шаг 2: Предсказание цены для новых объектов")
input_mode = st.radio("Выберите способ ввода данных", ["CSV файл", "Ввод вручную"])

if input_mode == "CSV файл":
    pred_file = st.file_uploader("Загрузите CSV для предсказания", type="csv", key="pred")
    if pred_file and 'best_model' in locals():
        df_pred = pd.read_csv(pred_file)
        y_pred = best_model.predict(df_pred)
        df_pred['predicted_price'] = y_pred
        st.success("Предсказание выполнено!")
        st.dataframe(df_pred)
else:
    if 'best_model' in locals():
        st.subheader("Введите признаки автомобиля")

        year = st.slider("Год выпуска", 1990, 2025, 2020)
        km_driven = st.number_input("Пробег (км)", 0, 1000000, 50000)
        mileage = st.number_input("Пробег на литр", 1.0, 50.0, 15.0, step=0.1)
        engine = st.number_input("Объем двигателя (cc)", 500, 5000, 1500)
        max_power = st.number_input("Максимальная мощность (bhp)", 50, 1000, 100)
        fuel = st.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
        owner = st.selectbox("Количество владельцев", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9])

        input_dict = {
            'year': [year],
            'km_driven': [km_driven],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'seats': [seats]
        }

        input_df = pd.DataFrame(input_dict)

        if st.button("Предсказать цену"):
            price = predict_single(best_model, input_df)[0]
            st.success(f"Предсказанная цена автомобиля: {price:,.0f}")
