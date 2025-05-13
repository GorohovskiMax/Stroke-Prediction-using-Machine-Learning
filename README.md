# 🧠 Stroke Prediction with Machine Learning

**Predictive modeling and identification of key risk factors for stroke using machine learning**

This project analyzes a medical dataset from Kaggle and builds machine learning models to predict whether a patient is likely to experience a stroke. The analysis includes data preprocessing, model training, evaluation, and a discussion of key contributing features.

---

## 📦 Dataset

The dataset used in this project is the **Stroke Prediction Dataset** from Kaggle.

➡️ [Access the dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

⚠️ **Note:** Due to licensing and usage restrictions, the dataset is not included in this repository.  
If you wish to run this notebook yourself, please **download the dataset manually from Kaggle** and upload it to your own Colab or local environment.

---

## 📊 Project Objectives

- Build predictive models to assess stroke risk.
- Identify key risk factors using feature importance analysis.
- Compare performance between different machine learning classifiers.

---

## 🔧 Project Structure

### 1️⃣ Data Preprocessing
- Handling missing values (e.g., BMI)
- Encoding categorical features
- Group-based imputation strategies

### 2️⃣ Exploratory Data Analysis (EDA)
- Visual exploration of age, gender, health conditions, and stroke occurrences
- Insights into class imbalance and distribution

### 3️⃣ Model Development
- **Logistic Regression**
- **Decision Tree (CART - Gini Impurity)**

Both models were trained using a pipeline including:
- Feature scaling
- SMOTE (Synthetic Minority Over-sampling Technique)
- GridSearchCV for hyperparameter tuning

### 4️⃣ Evaluation
- Accuracy, precision, recall, F1-score
- ROC-AUC score
- Visualizations for model performance and decision paths

---

## 📚 Key Learnings

- Hands-on experience with imputation and preprocessing medical data
- Training and evaluating binary classifiers for health prediction
- Understanding the trade-offs between precision and recall in imbalanced datasets
- Visual interpretation of decision trees for model transparency

---

## 🛠 Technologies Used

- Python
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- imblearn (for SMOTE)
- Jupyter Notebook / Google Colab

---

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
2. Upload it to your runtime environment (e.g., Google Colab)
3. Open and run the notebook cell
