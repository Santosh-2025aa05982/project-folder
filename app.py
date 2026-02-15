import streamlit as st 
import pandas as pd 
import pickle 
from sklearn.metrics import confusion_matrix, classification_report 
import seaborn as sns 
import matplotlib.pyplot as plt 

st.title("Student Performance Prediction Dashboard")
upload_file - st.file_uploader("Upload Test Data (CSV)", type = "csv")

if uploaded_file:
  data = pd.read_csv(uploaded_file)
  st.wrire("Data Preview:", data.head())

#MFeature: Model selection
model_option = st.selectbox(
  "Choose a Classification Model",
  ("logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", 'xgboost")
  )

  #Load model 
  with open(f'model/{model_options}.pkl', 'rb') as f:
    model = pickle.load(f)

#Assuming data is preprocessed like the training set 
if st.button("Run Evaluation"):
  
