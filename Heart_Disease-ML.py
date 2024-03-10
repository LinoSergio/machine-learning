# Data management libraries

import pandas as pd
import numpy as np

# Libraries for EDA

import matplotlib.pyplot as plt
import seaborn as sns

# Loading datasets and featuring engeniering

df = pd.read_csv('/Users/linosergiorocha/Projetos/Datasets/Heart_Disease_Prediction.csv.xls')
df

# Feature engeniering (Target variable as factor)

df['Heart Disease'] = (df['Heart Disease'] == 'Presence').astype(int)
df.head()

# Data Visualization

df.columns

# Analyzing continous and categorial variables separately

cat = df[['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']]
cont = df[['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']]

# Visualizing all unique values in the dataframe

for i in df.columns:
  print(f"{i}: {df[i].nunique} ")



for i in df.columns:
  plt.figure(figsize= (10,6))
  plt.title(f"Coluna avaliada: {i}", fontsize = 12)
  if i in cat:
    sns.countplot(x = df[i], hue = df['Heart Disease'])
  if i in cont:
    sns.histplot(df[i], kde = True)
    

corr = df.corr()
sns.heatmap(corr, annot= True, cmap= "Blues", fmt= ".1f", linewidths= .5)

## Using profile report

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title = "Heart Disease database")
profile.to_notebook_iframe()

# Preprocessing data

df1 = pd.get_dummies(df, columns=['Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'])

# Data preparation

X = df1.drop('Heart Disease', axis=1)
X

y = df1['Heart Disease']
y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .3, random_state= 0)

## Logistic Regrssion

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(X_train, y_train)

y_predict = log.predict(X_test)
y_predict

from sklearn.metrics import accuracy_score,r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

log.score(X_train,y_train)
log.score(X_test,y_test)

roc_auc_score(y_test, y_predict)

# Confusion Matrix

cm = confusion_matrix(y_test,y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= log.classes_).plot()

# ROC Curve

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_predict)