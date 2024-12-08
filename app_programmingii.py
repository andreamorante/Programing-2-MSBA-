import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load and clean data
def load_data():
    s = pd.read_csv("social_media_usage.csv")
    ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()

    # Rename columns
    ss.rename(columns={
        'web1h': 'sm_li',
        'educ2': 'education',
        'par': 'parent',
        'marital': 'married',
        'gender': 'female'
    }, inplace=True)

    # Apply transformations
    ss['sm_li'] = ss['sm_li'].apply(lambda x: 1 if x == 1 else 0)
    ss['married'] = ss['married'].apply(lambda x: 1 if x == 1 else 0)
    ss['female'] = ss['female'].apply(lambda x: 1 if x == 2 else 0)
    ss['parent'] = ss['parent'].apply(lambda x: 1 if x == 1 else 0)

    # Remove rows with values out of range or missing data
    ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)].dropna()

    return ss

data = load_data()

# Prepare data for training
X = data[['income', 'education', 'parent', 'married', 'female', 'age']]
y = data['sm_li']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Labels for education and income categories
education_labels = {
    1: 'Less than high school',
    2: 'High school incomplete',
    3: 'High school graduate',
    4: 'Some college, no degree',
    5: 'Two-year associate degree',
    6: 'Four-year college degree',
    7: 'Some postgraduate schooling',
    8: 'Postgraduate or professional degree'
}
income_labels = {
    1: '<$10k',
    2: '$10k-$20k',
    3: '$20k-$30k',
    4: '$30k-$40k',
    5: '$40k-$50k',
    6: '$50k-$75k',
    7: '$75k-$100k',
    8: '$100k-$150k',
    9: '$150k+'
}

# Streamlit app 
st.title("LinkedIn User Prediction")
st.write("Enter the following details:")

# User inputs
age = st.number_input("Age", min_value=10, max_value=98, value=30)

# Use descriptive labels for education and income
education_label = st.selectbox("Education Level", options=list(education_labels.values()), index=2)
education = list(education_labels.keys())[list(education_labels.values()).index(education_label)]

income_label = st.selectbox("Income Level", options=list(income_labels.values()), index=4)
income = list(income_labels.keys())[list(income_labels.values()).index(income_label)]

parent = st.selectbox("Are you a parent?", options=["Yes", "No"])
married = st.selectbox("Marital Status", options=["Married", "Not Married"])
gender = st.selectbox("Gender", options=["Female", "Male/Other"])

# Map other inputs to encoded format (yes or no)
parent_encoded = 1 if parent == "Yes" else 0
married_encoded = 1 if married == "Married" else 0
female_encoded = 1 if gender == "Female" else 0

# Create input array for prediction
user_input = np.array([[income, education, parent_encoded, married_encoded, female_encoded, age]])

# Predict using the model
prediction = model.predict(user_input)
probability = model.predict_proba(user_input)

# Display results
st.write(f"Prediction: {'LinkedIn User' if prediction[0] == 1 else 'Not a LinkedIn User'}")
st.write(f"Probability of being a LinkedIn User: {probability[0][1]:.2f}")