# Credit Card Fraud Detection using Logistic Regression

This project aims to detect fraudulent credit card transactions using a **Logistic Regression** model. The dataset is highly imbalanced, making the detection of fraud a challenging task. The model leverages basic features from the dataset, such as `V1`, `V2`, `V3`, and `Amount`, to make predictions on whether a transaction is legitimate or fraudulent.

## 🚀 Project Overview

- **Logistic Regression** model to predict fraudulent transactions.
- Handling of imbalanced class distribution by using `class_weight='balanced'`.
- Evaluation through **classification report**, **confusion matrix**, and **precision-recall curve**.
- **Streamlit** app for interactive fraud detection using the trained model.

## 🧑‍💻 Requirements

To run this project, you’ll need to install the following dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib streamlit
```

📊 Features

Class Distribution Visualization: Shows the imbalance in the dataset.

Model Training: Logistic regression trained on selected features from the dataset.

Evaluation: Includes classification report, confusion matrix, and precision-recall curve for model performance.

Streamlit App: A simple web interface to input transaction details and detect fraud in real-time.

📂 Files

train_model.py: Script to train the logistic regression model and save it as model.pkl.

creditcard.csv: The dataset containing credit card transactions.

model.pkl: The trained logistic regression model.

class_distribution.png: Visualization of class distribution.

confusion_matrix.png: Confusion matrix of model evaluation.

precision_recall_curve.png: Precision-recall curve of model performance.

app.py: Streamlit app to input transaction details and detect fraud in real-time.

o Run
Train the model: Run the following command to train the logistic regression model and save it as model.pkl:
```bash
python train_model.py
```
This will train the model and save it as model.pkl in your directory.

Start the Streamlit app: After training, you can start the Streamlit app to check if a transaction is fraudulent:

```bash
streamlit run app.py
```
This will open a web interface where you can enter transaction details to check if they are fraudulent.

 Evaluation
Due to the highly imbalanced data, the logistic regression model’s performance is suboptimal, with the model struggling to detect fraudulent transactions effectively. Evaluation metrics such as precision-recall AUC and confusion matrix provide insights into the model's weaknesses.
