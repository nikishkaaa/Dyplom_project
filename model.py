import numpy as np
import streamlit as st
import pandas as pd
import time
import shap
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from process_bank_churn import preprocess_new_data
import joblib
from sklearn.compose import ColumnTransformer
from joblib import load
from sqlalchemy import create_engine
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from sklearn.metrics import precision_recall_curve
from sklearn.inspection import PartialDependenceDisplay


@st.cache_resource
def load_model():
    return joblib.load('model/best_xgb.joblib')
model = load_model()


@st.cache_resource
def load_preprocessor():
    return load('model/preprocessor.joblib')

preprocessor = load_preprocessor()


def categorise(categorical_cols: list, df):
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df


def predict(input_data):
    numeric_cols = [
        'Age',
        'Balance',
        'CreditScore',
        'IsActiveMember',
        'EstimatedSalary', 
        'NumOfProducts', 
        'Tenure'
    ]
    categorical_cols = ['Geography', 'Gender']

    data = preprocess_new_data(input_data, preprocessor)
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1][0]
    return prediction[0], probability



engine = create_engine("postgresql://postgres:tvremtoh@localhost:5432/Churn")
query = "SELECT * FROM customer;"
data = pd.read_sql(query, engine)
data = data.drop('bank_id', axis = 1)


balance_max = data['Balance'].max()
credit_score_max = data['CreditScore'].max()
estimated_salary_max = data['EstimatedSalary'].max()
num_of_products_max = data['NumOfProducts'].max()
tenure_min = data['Tenure'].min()
tenure_max = data['Tenure'].max()


X_train = pd.read_csv('data/X_train.csv')
X_val = pd.read_csv('data/X_val.csv')
y_train = pd.read_csv('data/y_train.csv')
y_val = pd.read_csv('data/y_val.csv')


def display_prediction():

    st.markdown("<h3 style='text-align: center;'>Введіть дані про клієнта</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Вік клієнта', min_value=18, max_value=100, value=30, step=1)
        balance = st.number_input('Баланс на рахунку', min_value=0.0, max_value=balance_max+100000, value=50000.0, step=100.0)
        credit_score = st.number_input('Кредитний рейтинг', min_value=0, max_value=credit_score_max, value=650, step=1)
    
    with col2:
        is_active = st.selectbox('Клієнт активний?', ['Так', 'Ні'])
        estimated_salary = st.number_input('Орієнтовна зарплата', min_value=0.0, max_value=estimated_salary_max+100000, value=50000.0, step=100.0)
        geography = st.selectbox('Країна', ['France', 'Spain', 'Germany'])
        tenure = st.number_input('Кількість років, скільки клієнт є в банку', min_value=tenure_min, max_value=tenure_max+10, value=5, step=1)
    
    with col3:
        gender = st.selectbox('Стать клієнта', ['Чоловік', 'Жінка'])
        num_of_products = st.slider('Кількість банківських продуктів', min_value=0, max_value=num_of_products_max+100, value=1)
        has_cr_card = st.selectbox('Має кредитну картку?', ['Так', 'Ні'])
        

    is_active_numeric = 1 if is_active == 'Так' else 0
    has_cr_card_numeric = 1 if has_cr_card == 'Так' else 0
    gender_map = {'Жінка': 'Female', 'Чоловік': 'Male'}
    gender_converted = gender_map.get(gender, gender)


    data = pd.DataFrame({
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'IsActiveMember': [is_active_numeric],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography],
    'Gender': [gender_converted],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_numeric], 
    'Tenure': [tenure]
    })

    
    if st.button("Прогноз відтока"):
        with st.spinner('Прогнозування...'):
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            prediction, probability = predict(data)
    
        st.markdown(f"<div class='prediction-result'>Чи залишиться людина клієнтом банку? {'Ні' if prediction == 1 else 'Так'}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='probability-result'>Ймовірність прогнозу: {probability:.2f}</div>", unsafe_allow_html=True)


    st.divider()
    st.markdown("<h3 style='text-align: center;'>Інформація про роботу моделі</h3>", unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        importance = model.get_booster().get_score(importance_type='weight')
        importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
        fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance', labels={'Importance': 'Feature Importance'})
        fig.update_layout(xaxis_title='Features', yaxis_title='Importance', showlegend=False)
        
        st.plotly_chart(fig)
    
    with a2: 
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess'))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True
        )
        
        st.plotly_chart(fig)

    st.divider()
    b1, b2 = st.columns(2)
    with b1: 
        cm = confusion_matrix(y_val, model.predict(X_val))
    
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Not Churned', 'Churned'],
            y=['Not Churned', 'Churned'],
            colorscale='Blues',
            showscale=True
        )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        st.plotly_chart(fig)

    with b2:
        precision, recall, _ = precision_recall_curve(y_val, y_prob)
        baseline_precision = float(y_val.mean())  # теперь это float
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='royalblue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=[min(recall), max(recall)],
            y=[baseline_precision, baseline_precision],
            mode='lines',
            name=f'Baseline (Precision = {baseline_precision:.2f})',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, 1.0]),
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig)




    