import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("🪻 Iris Dataset Explorer")

@st.cache_data
def load_data():
  iris = load_iris()
  df = pd.DataFrame(iris.data, columns = iris.feature_names)
  df['species'] = iris.target
  return df, iris.target_names

df, target_names = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

st.sidebar.title("Input Features")

sepal_length = st.sidebar.slider(
    label="Sepal Length (cm)",
    min_value=float(df["sepal length (cm)"].min()),
    max_value=float(df["sepal length (cm)"].max()),
    
)
sepal_width = st.sidebar.slider(
    label="Sepal Width (cm)",
    min_value=float(df["sepal width (cm)"].min()),
    max_value=float(df["sepal width (cm)"].max()),
    
)
petal_length = st.sidebar.slider(
    label="Petal Length (cm)",
    min_value=float(df["petal length (cm)"].min()),
    max_value=float(df["petal length (cm)"].max()),
    
)
petal_width = st.sidebar.slider(
    label="Petal Width (cm)",
    min_value=float(df["petal width (cm)"].min()),
    max_value=float(df["petal width (cm)"].max()),
    
)

input_data = [[sepal_length,sepal_width,petal_length, petal_width]]

## Prediction
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

st.write("prediction")
st.write(f"The predicted species is: {predicted_species}")
