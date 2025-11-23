import streamlit as st
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import plotly.express as px

model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

st.title("Dashboard de Clasificación de Especies de Iris")
st.markdown("Santiago Del Valle, David de la Hoz, Isabella Gomez")  

st.header("Métricas del Modelo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precisión", f"{metrics['accuracy']:.2f}")
col2.metric("Precisión (Precision)", f"{metrics['precision']:.2f}")
col3.metric("Sensibilidad (Recall)", f"{metrics['recall']:.2f}")
col4.metric("F1-Score", f"{metrics['f1']:.2f}")
st.write(f"**Media de Validación Cruzada:** {metrics['cv_mean']:.2f}")
st.text("Informe de Clasificación:")
st.code(metrics['classification_report'])

st.header("Predecir Especie de Iris")
st.write("Ingresa las medidas para predecir la especie y visualizar en 3D.")

sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, value=5.1)
sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, value=1.4)
petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, value=0.2)

if st.button("Predecir"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    species = iris.target_names[prediction[0]]
    st.success(f"Especie Predicha: **{species}**")
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(scaler.transform(iris.data))
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['species'] = iris.target_names[iris.target]
    
    new_point_pca = pca.transform(input_scaled)
    new_df = pd.DataFrame(new_point_pca, columns=['PC1', 'PC2', 'PC3'])
    new_df['species'] = species
    
    combined_df = pd.concat([df_pca, new_df], ignore_index=True)
    fig = px.scatter_3d(combined_df, x='PC1', y='PC2', z='PC3', color='species', 
                        title="Gráfico de Dispersión 3D (PCA) - Dataset + Nueva Muestra",
                        color_discrete_map={'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'})
    st.plotly_chart(fig)

st.header("Visualizaciones Adicionales")
st.subheader("Histogramas por Especie")
feature = st.selectbox("Selecciona Característica", iris.feature_names)
fig, ax = plt.subplots()
for i, species in enumerate(iris.target_names):
    sns.histplot(df[df['species'] == species][feature], ax=ax, label=species, kde=True)
ax.legend()
st.pyplot(fig)

st.subheader("Matriz de Dispersión")
fig = sns.pairplot(df, hue='species', diag_kind='kde')
st.pyplot(fig)