# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page config
st.set_page_config(page_title="BTC Price Direction Predictor", layout="wide")

# Title
st.title("📊 Logistic Regression App for BTC Price Direction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Raw Data Preview")
    st.dataframe(df.head())

    # Data Preprocessing
    df.drop(columns=["Date"], inplace=True, errors='ignore')
    df.dropna(inplace=True)

    df['Target'] = df['Change %'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['Change %'], inplace=True, errors='ignore')

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=200, C=0.1)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("2. Model Performance")
    st.markdown(f"**Accuracy:** {acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("3. Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("4. Feature Importance")
    importance = model.coef_[0]
    features = X.columns

    fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
    sns.barplot(x=importance, y=features, ax=ax_feat)
    ax_feat.set_title("Feature Coefficients")
    st.pyplot(fig_feat)

    st.subheader("5. Try Your Own Prediction")
    input_data = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success(f"📈 Prediction: {'Up' if prediction == 1 else 'Down'}")
else:
    st.info("Please upload a CSV file with columns including 'Change %', and optionally 'Date'.")
