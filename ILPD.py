import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('ILPD.csv')
    data.columns = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Selector']
    data['A/G Ratio'].fillna(data['A/G Ratio'].mean(), inplace=True)
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    return data

data = load_data()

# Train the models
@st.cache_resource
def train_models(data):
    X = data.drop('Selector', axis=1)
    y = data['Selector'].apply(lambda x: 1 if x == 1 else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    
    return rf_classifier, log_reg, scaler

rf_model, log_reg_model, scaler = train_models(data)

# Streamlit app
st.title('Liver Disease Prediction App')

# Sidebar for navigation
page = st.sidebar.selectbox('Choose a page', ['Prediction', 'Data Exploration'])

if page == 'Prediction':
    st.header('Predict Liver Disease')
    
    # Model selection
    model_choice = st.radio("Choose a model", ('Random Forest', 'Logistic Regression'))
    
    # Input fields
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    tb = st.number_input('Total Bilirubin', min_value=0.0, value=1.0)
    db = st.number_input('Direct Bilirubin', min_value=0.0, value=0.5)
    alkphos = st.number_input('Alkaline Phosphotase', min_value=0, value=128)
    sgpt = st.number_input('Sgpt', min_value=0, value=20)
    sgot = st.number_input('Sgot', min_value=0, value=30)
    tp = st.number_input('Total Proteins', min_value=0.0, value=3.9)
    alb = st.number_input('Albumin', min_value=0.0, value=1.9)
    ag_ratio = st.number_input('A/G Ratio', min_value=0.0, value=0.95)
    
    # Predict button
    if st.button('Predict'):
        gender_encoded = 1 if gender == 'Male' else 0
        new_data = pd.DataFrame([[age, gender_encoded, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]],
                                columns=['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio'])
        new_data_scaled = scaler.transform(new_data)
        
        if model_choice == 'Random Forest':
            prediction = rf_model.predict(new_data_scaled)
        else:
            prediction = log_reg_model.predict(new_data_scaled)
        
        if prediction[0] == 1:
            st.warning('The patient may have liver disease.')
        else:
            st.success('The patient does not have liver disease.')

elif page == 'Data Exploration':
    st.header('Data Exploration')
    
    # Display raw data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
    
    # Gender distribution
    st.subheader('Gender Distribution')
    gender_dist = data['Gender'].value_counts()
    st.bar_chart(gender_dist)
    
    # Age distribution
    st.subheader('Age Distribution')
    fig, ax = plt.subplots()
    data['Age'].hist(bins=20, ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader('Correlation Matrix')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Feature distributions
    st.subheader('Feature Distributions')
    feature = st.selectbox('Select a feature', data.columns[2:-1])
    fig, ax = plt.subplots()
    data[feature].hist(bins=20, ax=ax)
    ax.set_title(f'{feature} Distribution')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    st.pyplot(fig)

st.sidebar.info('This app predicts the likelihood of liver disease based on various health parameters.')