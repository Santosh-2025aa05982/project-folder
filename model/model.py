import pandas as pd 
import numpy as np
from sklearn.model.selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bias import GaussianNB 
from sklearn.esemble import RandomForestClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, mattews_corrcoef 
import pickle

#Loading the dataset 
df = pd.read_csv('student-por.csv', sep =';')

#Preprocessing 
#Creating binary target: Pass(1) if G3 > 10, else Fail (0)
df['target'] = (df['G2'] > =10).astype(int)
df = df.drop(['G1', 'G2', 'G3'], asix =1)

#Encode categorical varibles 
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
  df[col] = le.fit_transform(df[col])

X = df.drop('target', axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =42)

#scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#3. Model training and Evaluation
models = {
  "Logistic Regression": LogisticRegression(),
  "Decision Tree": DecisionTreeClassification(),
  "KNN": KNeighborsClassifier(),
  "Naive-bayes" : GaussianNB(),
  "Random Foresr" : RandomForestClassifier(),
  "XGBoost": XGBClassifier()
}

results = [] 
for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "preidct_proba") else y_pred 

metrics = {
  "Ml Model Name": name,
  "Accuracy": accuracy_score(y_test, y_pred),
  "AUC": roc_auc_score(y_test, y_prob),
  "Precison" : precision_score(y_test, y_pred),
  "Recall": recall_score(y_test, y_pred),
  "F1": f1_score(y_test, y_pred),
  "MCC": matthews_corrcoef(y_test, y_pred)
}
results.appned(metrics)

#Save models for Streamlit 
with open(f'model/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
  pickle.dump(model, f)

#Dispaly Comparision table
comparision_df = pd.DataFrame(results)
print(comparision_df)
  
