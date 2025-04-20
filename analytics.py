import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import altair as alt
from streamlit_extras.dataframe_explorer import dataframe_explorer
from sqlalchemy import create_engine
import plotly.express as px


def display_analytics():
    engine = create_engine("postgresql://postgres:tvremtoh@localhost:5432/Churn")
    query = "SELECT * FROM customer;"
    data = pd.read_sql(query, engine)
    data = data.drop('bank_id', axis = 1)

    query2 = "SELECT * FROM category;"
    category = pd.read_sql(query2, engine)
    data = data.merge(category, on = 'id', how = 'inner')
    
    ##st.dataframe(data, use_container_width = True)
    ##st.sidebar.image('')
    
    with st.sidebar:
        st.title('Фільтри')
        
        min_income, max_income = st.slider(
            'Діапазон доходу (у грн):',
            min_value=0,
            max_value=100000,
            value=(20000, 50000),
            step=1000
        )
    
        st.success(f'Ви вибрали аналітику для доходу від {min_income} грн до {max_income} грн')


    df2 = data[(data['EstimatedSalary'] >= min_income) & (data['EstimatedSalary'] <= max_income)]
    with st.expander('Фільтрація датасету'):
        filtered_df = dataframe_explorer(df2, case = False)
        st.dataframe(filtered_df, use_container_width = True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("<h3 style='text-align: center;'>Відтік клієнтів</h3>", unsafe_allow_html=True)
        active_counts = df2['prediction'].value_counts().reset_index()
        active_counts.columns = ['prediction', 'Count']
        active_counts['prediction'] = active_counts['prediction'].map({1: 'Пішов', 0: 'Залишився'})
        fig = px.pie(active_counts, names='prediction', values='Count', hole=0.4, 
                     color_discrete_sequence= ['#ab1b31', '#8a8384'])
        st.plotly_chart(fig, use_container_width=True)

    
    with a2:
        st.markdown("<h3 style='text-align: center;'>Метрики датасету</h3>", unsafe_allow_html=True)
        from streamlit_extras.metric_cards import style_metric_cards
        col1, col2, = st.columns(2)
        col1.metric(label = 'Загальна к-сть екземплярів', value = df2.id.count(), delta = 'Загальна к-сть екземплярів')
        col2.metric('Загальна  к-сть куплених продуктів', value = f"{df2.NumOfProducts.sum():,.0f}", delta = df2.NumOfProducts.median())

        col11, col22, col33 = st.columns(3)
        col11.metric(label = 'Max заробітня плата', value = df2.EstimatedSalary.max(), delta = 'Max заробітня плата')
        col22.metric(label = 'Найкращий кредитний рейтинг', value = df2.EstimatedSalary.max(), delta = 'Найкращий рейтинг')
        col33.metric(label = 'Найбільший баланс', value = df2.Balance.max(), delta = 'Найбільший баланс')
        style_metric_cards(border_left_color = '#23b049', border_color = '#0c0d0d')

    st.divider()
    
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("<h3 style='text-align: center;'>Країна & Заробітня плата</h3>", unsafe_allow_html=True)
        source = df2 
        chart = alt.Chart(source).mark_circle().encode(
            x = 'Geography', 
            y = 'EstimatedSalary', 
            color = 'Geography'
        ).interactive()
        st.altair_chart(chart, theme = 'streamlit', use_container_width = True)
        

    with b2:
        st.markdown("<h3 style='text-align: center;'>Гендер & Заробітня плата</h3>", unsafe_allow_html=True)
        source2 = pd.DataFrame(
            {
                'Gender': df2['Gender'], 
                'EstimatedSalary': df2['EstimatedSalary'], 
                'Geography': df2['Geography']
            }
        ) 
        
        bar_chart = alt.Chart(source2).mark_bar().encode(
            x = 'Geography', 
            y = 'EstimatedSalary', 
            color = 'Gender'
        )
        st.altair_chart(bar_chart, use_container_width = True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h3 style='text-align: center;'>Вік клієнтів & Відтік</h3>", unsafe_allow_html=True)
        fig = px.histogram(df2, x='Age', color='prediction',
                           nbins=30, barmode='stack',
                           color_discrete_sequence= ['#ab1b31', '#8a8384'])
        fig.update_layout(xaxis_title='Вік', yaxis_title='Кількість')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<h3 style='text-align: center;'>Активність клієнтів</h3>", unsafe_allow_html=True)
        active_counts = df2['IsActiveMember'].value_counts().reset_index()
        active_counts.columns = ['IsActiveMember', 'Count']
        active_counts['IsActiveMember'] = active_counts['IsActiveMember'].map({1: 'Активні', 0: 'Неактивні'})
        fig = px.pie(active_counts, names='IsActiveMember', values='Count', hole=0.4, 
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

    
    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("<h3 style='text-align: center;'>Кредитний рейтинг vs Відтік</h3>", unsafe_allow_html=True)
        fig = px.box(df2, x='prediction', y='CreditScore', points='all',
                     color='prediction',
                     labels={'prediction': 'Статус (0 = залишився, 1 = пішов)', 'CreditScore': 'Кредитний рейтинг'},
                     color_discrete_sequence= ['#8bbbf7', '#8a8384'])
        st.plotly_chart(fig, use_container_width=True)
    
    with d2:
        st.markdown("<h3 style='text-align: center;'>Теплова карта кореляції</h3>", unsafe_allow_html=True)
        corr = df2[['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'prediction']].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)





