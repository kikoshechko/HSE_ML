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
st.title("Предсказание цены автомобиля")

# 1. Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV с данными", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Загрузите CSV файл или используйте пример данных.")
    df = pd.DataFrame()

# 2. EDA
if not df.empty:
    st.subheader("Обзор данных")
    st.write(df.head())

    st.subheader("Основные гистограммы признаков")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

# 3. Настройка признаков
cat_features = ["fuel", "seller_type", "transmission", "owner", "seats"]
num_features = ["year", "km_driven", "mileage", "engine", "max_power"]
target = "selling_price"

if not df.empty and target in df.columns:
    X = df.drop(columns=[target, "name"], errors="ignore")
    y = df[target]

    # Предобработка
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    # Модель
    pipe = Pipeline([("preprocessor", preprocessor), ("model", Ridge())])

    param_grid = {"model__alpha": np.logspace(-4, 2, 10)}
    grid = GridSearchCV(pipe, param_grid, cv=10, scoring="r2", n_jobs=-1)

    with st.spinner("Обучение модели..."):
        grid.fit(X, y)

    best_model = grid.best_estimator_

    # Метрики
    y_pred = best_model.predict(X)
    st.subheader("Результаты обучения")
    st.write("Лучший alpha:", grid.best_params_["model__alpha"])
    st.write(f"R²: {r2_score(y, y_pred):.4f}")
    st.write(f"MSE: {MSE(y, y_pred):.2f}")

    # 4. Визуализация весов модели

    st.subheader("Визуализация весов модели")
    feature_names_num = num_features
    feature_names_cat = (
        best_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(cat_features)
    )
    all_features = np.concatenate([feature_names_num, feature_names_cat])
    coef = best_model.named_steps["model"].coef_

    coef_df = pd.DataFrame({"feature": all_features, "coef": coef})
    coef_df = coef_df.sort_values(by="coef", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="coef", y="feature", data=coef_df, ax=ax)
    ax.set_title("Веса признаков")
    st.pyplot(fig)

# 5. Предсказание для новых объектов
st.subheader("Предсказание цены для новых объектов")
input_method = st.radio("Выберите способ ввода данных", ["CSV файл", "Ввод вручную"])

if input_method == "CSV файл":
    uploaded_input = st.file_uploader(
        "Загрузите CSV для предсказания", type=["csv"], key="input_csv"
    )
    if uploaded_input is not None and best_model is not None:
        new_data = pd.read_csv(uploaded_input)
        pred = best_model.predict(new_data)
        st.write(pd.DataFrame({"predicted_price": pred}))

elif input_method == "Ввод вручную":
    if best_model is not None:
        st.write("Введите значения признаков:")
        new_obj = {}
        for f in num_features:
            new_obj[f] = st.number_input(f)
        for f in cat_features:
            new_obj[f] = st.selectbox(
                f, df[f].unique() if not df.empty else ["Option1", "Option2"]
            )

        if st.button("Предсказать цену"):
            new_df = pd.DataFrame([new_obj])
            pred = best_model.predict(new_df)
            st.write(f"Предсказанная цена: {pred[0]:.2f}")
