import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error as MSE

st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Sidebar
st.sidebar.title("Режим работы")
mode = st.sidebar.radio("Выберите режим:", ["EDA", "Предсказание цены"])

# Helper functions 
def build_model(df):
    target = 'selling_price'
    
    cat_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    num_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']

    X = df.drop(columns=[target, 'name'], errors='ignore')
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    param_grid = {'model__alpha': np.logspace(-4, 2, 10)}

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    
    return grid.best_estimator_, num_features, cat_features

def plot_weights(model, num_features, cat_features):
    coefs = model.named_steps['model'].coef_
    cat_transformer = model.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = cat_transformer.get_feature_names_out(cat_features)
    all_features = list(num_features) + list(cat_names)
    
    coef_df = pd.DataFrame({'feature': all_features, 'coefficient': coefs})
    coef_df = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=False).index)
    
    st.subheader("Веса признаков модели")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='coefficient', y='feature', data=coef_df, ax=ax)
    st.pyplot(fig)

# Mode: EDA
if mode == "EDA":
    st.title("EDA - Исследование данных автомобилей")
    uploaded_file = st.file_uploader("Загрузите CSV с данными", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        st.subheader("Гистограммы числовых признаков")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

        st.subheader("Корреляционная матрица")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Mode: Prediction
else:
    st.title("Предсказание цены автомобиля")
    input_method = st.radio("Выберите способ ввода данных", ["CSV файл", "Ввод вручную"])
    
    if input_method == "CSV файл":
        pred_file = st.file_uploader("Загрузите CSV для предсказания", type="csv")
        df_pred = pd.read_csv(pred_file) if pred_file else pd.DataFrame()
    
    else:
        st.subheader("Введите признаки автомобиля")
        df_pred = pd.DataFrame({
            'year': [st.number_input("Год выпуска", 1990, 2025, 2020)],
            'km_driven': [st.number_input("Пробег (км)", 0, 1000000, 50000)],
            'mileage': [st.number_input("Пробег на литр", 0.0, 50.0, 15.0)],
            'engine': [st.number_input("Объем двигателя (cc)", 500, 10000, 1500)],
            'max_power': [st.number_input("Максимальная мощность (bhp)", 10, 1000, 100)],
            'fuel': [st.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])],
            'seller_type': [st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])],
            'transmission': [st.selectbox("Коробка передач", ["Manual", "Automatic"])],
            'owner': [st.selectbox("Количество владельцев", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])],
            'seats': [st.number_input("Количество мест", 1, 20, 5)],
        })
    
    st.subheader("Обучение модели на примере данных")
    train_file = st.file_uploader("Загрузите CSV с обучающими данными", type="csv")
    
    if train_file is not None and not df_pred.empty:
        df_train = pd.read_csv(train_file)
        model, num_features, cat_features = build_model(df_train)
        st.success("Модель обучена!")
        
        plot_weights(model, num_features, cat_features)
        
        st.subheader("Предсказания")
        X_new = df_pred.drop(columns=['selling_price', 'name'], errors='ignore')
        df_pred['predicted_price'] = model.predict(X_new)
        st.dataframe(df_pred)
