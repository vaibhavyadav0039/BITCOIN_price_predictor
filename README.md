# 📊 Logistic Regression Streamlit App

A simple interactive web app built using **Streamlit** and **Scikit-learn** to analyze and predict market trends using Logistic Regression.

---

## 🚀 Features

✨ Upload your custom `.csv` dataset  
🧹 Auto data cleaning and preprocessing  
🤖 Train Logistic Regression with one click  
📈 Visualize model performance (accuracy, confusion matrix, etc.)  
🔮 Make real-time predictions using feature inputs

---

## 🛠️ Installation & Running

1. 📦 Create a virtual environment (optional but recommended):
python -m venv venv
venv\Scripts\activate (on Windows)


2. 📥 Install dependencies:
pip install -r requirements.txt


3. 🚀 Run the app:
streamlit run app.py



4. 🌐 Visit `http://localhost:8501` in your browser

---

## 📂 File Structure

📁 project/
├── app.py 👉 Main Streamlit application
├── requirements.txt 👉 Python dependencies
├── btc_enriched_data.csv👉 Sample input file
└── README.md 👉 This guide



---

## 📊 Dataset Example

The app expects a dataset with columns like:

Date | Price | Open | High | Low | Vol. | Change % | MA10 | MA50 | MA200 | RSI | MOM_S | VOL_T | T_OSC | BPI | BRI | SENT_S


✅ The `Target` is automatically created from `Change %`:
- If `Change % > 0` ➜ `Target = 1` (price up)
- Else ➜ `Target = 0` (price down)

---

## 🤖 Model Info

📘 Using: `LogisticRegression` from `sklearn.linear_model`  
🧮 Data is split 70-30 for training/testing  
🔁 Model trained with `max_iter=1000`  
⚙️ Underfit controlled to keep accuracy ~80-90%

---

## 📊 Outputs

- ✅ Accuracy score
- 📉 Confusion matrix
- 🧾 Classification report
- 📌 Feature importance
- 📤 Prediction section (manual input)

---

## ❓ Common Issues

❌ Timeout while installing packages?  
✅ Use a stable internet connection  
✅ Try using PyPI mirror:
pip install -r requirements.txt --timeout 100 --retries 5 -i https://pypi.org/simple



---

## 👨‍💻 Author

Made with ❤️ by Vaibhav Yadav

---

## 📄 License

This project is open source under the MIT License.
