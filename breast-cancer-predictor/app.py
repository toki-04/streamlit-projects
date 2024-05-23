import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import io


# Title
st.title("Breast Cancer Predictor")
st.text("V.0.0.1")

# DataFrame
df = pd.read_csv("./cancer.csv")
st.header("Breast Cancer DataFrame")
st.dataframe(df, use_container_width=True)
st.text("Breast Cancer DataFrame Shape: " + str(df.shape))

# DataFrame Info
st.header("Breast Cancer DataFrame Information")

buffer = io.StringIO()
df.info(buf=buffer)
df_info = buffer.getvalue()[37:] # remove text "<class 'pandas.core.frame.DataFrame'>"
st.text(df_info)

# DataFrame Statistics
st.header("Breast Cancer DataFrame Statistics")
st.dataframe(df.describe(), use_container_width=True)

# Breast Cancer Diagnosis Count
# Diagnosis Count Plot
st.header("Breast Cancer Diagnosis Count")

fig = plt.figure(figsize=(10, 4))
sns.countplot(df["diagnosis"], label="count")
st.pyplot(fig)

st.text("B - Benign  |  M - Malignant")
st.text(df["diagnosis"].value_counts())

# Label Encoder
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df.iloc[:, 1] = lb.fit_transform(df.iloc[:, 1].values)

# Breast Cancer DataFrame Heatmap
st.header("Breast Cancer DataFrame Heatmap")

plt = plt.figure(figsize=(10, 7))
sns.heatmap(df.iloc[:, 1:10].corr(), annot=True)
st.pyplot(plt)

# Breast Cancer DataFrame Pairplot
# st.header("Breast Cancer DataFrame Pairplot")

# plt = plt.figure(figsize=(10, 7))
# sns.pairplot(df.iloc[:, 1:5], hue="diagnosis")
# st.pyplot(plt)
# plot = plt.plot(diagnosis)


# Training
X = df.iloc[:, 2:32].values
y = df.iloc[:, 1].values

## Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

y_train = y_train.astype("int")
y_test = y_test.astype("int")

## Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

## Model
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

# Train
log.fit(X_train, y_train)

# Metrics
st.header("Metrics")
log_score = log.score(X_train, y_train)
st.text("LogisticRegression Score: "+str(log_score))

from sklearn.metrics import accuracy_score, classification_report
acc_score = accuracy_score(y_test, log.predict(X_test))
st.text("Accuracy Score: "+str(acc_score))

# classification_report
st.text(classification_report(y_test, log.predict(X_test)))













