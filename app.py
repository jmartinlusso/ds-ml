import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.title("Iris Classifier — Day 1")

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model_choice = st.selectbox("Pick a model", ["Logistic Regression", "Decision Tree"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
else:
    model = DecisionTreeClassifier(random_state=41)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.write(f"Accuracy on held-out test set: {accuracy:.2%}")