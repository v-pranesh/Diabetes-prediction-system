# 🩺 Diabetes Prediction System  

A web-based application built with **Streamlit** to predict the likelihood of a patient having diabetes using a machine learning model.  

---

## 🎯 About the Project  
This application provides a user-friendly interface for healthcare professionals or individuals to input patient data and receive an instant prediction on their diabetes status. The model is trained on the **Pima Indians Diabetes Dataset**, which contains medical diagnostic measurements for female patients.  

The **Random Forest Classifier** typically achieves an accuracy of approximately **75–80%** on this dataset.  

---

## ✨ Features  
- **Interactive Sidebar**: Input health metrics such as Pregnancies, Glucose, Blood Pressure, BMI, etc.  
- **Real-time Prediction**: Immediate result — **Diabetic** or **Not Diabetic**.  
- **Visualized Patient Report**: Comparative static scatter plots for each metric, showing where the user's data falls relative to the dataset. Generated using **Matplotlib** and **Seaborn**.  
- **Model Accuracy Display**: Shows the accuracy of the trained Random Forest Classifier on the test data.  

---

# 🚀 Installation

### Prerequisites
- Python 3.x installed on your system

### Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 👨‍💻 Usage
Run the application with:
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 File Structure
```
Diabetes-Prediction/
│── app.py          # Main Streamlit application script
│── diabetes.csv    # Dataset used for training
│── requirements.txt# Python dependencies
│── Procfile        # Deployment config for Heroku
│── setup.sh        # Streamlit setup script for deployment
```

---

## 📦 Dependencies
The project uses the following libraries (listed in `requirements.txt`):
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Pillow
- plotly

---

## 🧠 How It Works
1. **Data Loading**: Loads the `diabetes.csv` dataset.
2. **Data Preparation**: Splits the dataset into features (X) and target (y), then into training and test sets.
3. **Model Training**: A `RandomForestClassifier` is trained on the training data.  
   > Note: The model is retrained dynamically each time `app.py` is run.
4. **Prediction**: Based on user inputs, the model predicts whether the patient is **Diabetic** or **Not Diabetic**.
5. **Visualization & Reporting**: Displays the prediction result and comparative plots showing how the user’s values compare with the dataset.

---

## 📸 Screenshots
*(Add screenshots inside a `screenshots/` folder and link them here)*

### 🏠 Input Form
![Input Screenshot](screenshots/input.png)

### 📊 Prediction Result
![Prediction Screenshot](screenshots/prediction.png)

### 📉 Visualization Report
![Visualization Screenshot](screenshots/visualization.png)

---

## 📄 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
