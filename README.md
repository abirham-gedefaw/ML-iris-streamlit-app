# 🪻 Iris Species Classification App

A web-based machine learning application built with **Streamlit** and **Scikit-Learn** that predicts the species of an Iris flower based on its physical dimensions. 

Users can manipulate sepal and petal measurements via interactive sidebar sliders to see real-time machine learning classification.

## 🚀 Features
* **Interactive UI:** Dynamic sidebar sliders to control feature values (Sepal/Petal Length & Width).
* **Smart Performance:** Built using `@st.cache_data` to ensure dataset and model calculations load instantly without lag.
* **Robust ML:** Built on top of the classic 3-species Iris dataset (Setosa, Versicolor, Virginica).
  
## 🤖 Machine Learning Model

This application uses a **Random Forest Classifier** built with Scikit-Learn to evaluate the flower dimensions and predict the correct species. 

### Why Random Forest?
* **High Accuracy:** It builds multiple decision trees and merges them together to get a more accurate and stable prediction.
* **Feature Importance:** It handles the highly-correlated features of the Iris dataset (like petal length and width) exceptionally well without overfitting.
* **No Feature Scaling Required:** It can process raw centimeter values directly from the Streamlit sliders without needing normalization or standardization.

### Model Features (Inputs):
1. **Sepal Length** (cm)
2. **Sepal Width** (cm)
3. **Petal Length** (cm)
4. **Petal Width** (cm)

## 🛠️ Installation & Local Setup

Follow these steps to run the application on your computer:

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd iris-streamlit-app
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # On Windows activation:
   .\venv\Scripts\activate
   # On Mac/Linux activation:
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run classification.py
   ```

## 📦 Project Structure
```text
├── classification.py   # Main application script with layout and ML model
├── requirements.txt     # Python dependencies (Streamlit, Scikit-Learn, Pandas)
├── .gitignore          # Files excluded from GitHub tracking
└── README.md           # Project overview and instructions
```

## 📊 Dataset Information
The app utilizes the classic Scikit-Learn **Iris Dataset**, which includes 150 samples across 3 distinct flower classes:
* *Iris-Setosa*
* *Iris-Versicolor*
* *Iris-Virginica*
