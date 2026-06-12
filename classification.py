import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Classifier", layout="wide")
st.title("🪻 Iris Dataset Explorer & Classifier")

# 1. Cache the dataset loading
@st.cache_data
def get_iris_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    return df, iris

df, iris = get_iris_data()

# Train model once and cache it
@st.cache_resource
def train_model(dataframe):
    X = dataframe[iris.feature_names]
    y = dataframe['species']
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf

model = train_model(df)

# 2. Setup the Sidebar Inputs
st.sidebar.header("🔧 Adjust Flower Dimensions")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), 4.3)
petal_width = st.sidebar.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), 1.3)

# 3. Create prediction DataFrame to solve the UserWarning
input_data = pd.DataFrame([{
    "sepal length (cm)": sepal_length,
    "sepal width (cm)": sepal_width,
    "petal length (cm)": petal_length,
    "petal width (cm)": petal_width
}])

# Layout layout split: Left side for prediction details, Right side for charts
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔮 Prediction Results")
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    st.success(f"Predicted Species: **{prediction.upper()}**")
    
    # Show prediction probabilities in a clean table
    prob_df = pd.DataFrame({
        "Species": model.classes_,
        "Confidence (%)": [round(p * 100, 2) for p in probabilities]
    })
    st.dataframe(prob_df, hide_index=True)

with col2:
    st.subheader("📊 Visualizing Your Input Data Point")
    
    # Generate an interactive scatter plot
    fig = px.scatter(
        df, 
        x="petal length (cm)", 
        y="petal width (cm)", 
        color="species",
        title="Petal Length vs Width (The Red Target symbol represents YOUR selection)",
        color_discrete_map={"setosa": "#636EFA", "versicolor": "#EF553B", "virginica": "#00CC96"}
    )
    
    # Add a custom star marker exactly where the user's sliders currently sit
    fig.add_scatter(
        x=[petal_length], 
        y=[petal_width], 
        mode="markers", 
        marker=dict(size=15, color="red", symbol="x", line=dict(width=2, color="white")),
        name="Your Input"
    )
    
    st.plotly_chart(fig, use_container_width=True)