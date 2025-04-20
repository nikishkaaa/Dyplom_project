import numpy as np
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import time
from streamlit_option_menu import option_menu
from main_page import display_home_page
from analytics import display_analytics
from model import display_prediction
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

selected = option_menu(
        None, 
        ["Головна", "Графіки", "Прогноз"], 
        icons=["house", "bar-chart", "cpu"], 
        menu_icon="cast", 
        default_index=0, 
        orientation = "horizontal",
    )

if selected == "Головна":
    display_home_page() 

elif selected == "Графіки":
    st.markdown("<h2 style='text-align: center;'>📊 Графіки</h2>", unsafe_allow_html=True)
    display_analytics()

elif selected == "Прогноз":
    st.markdown("<h2 style='text-align: center;'>🧠 Прогноз</h2>", unsafe_allow_html=True)
    display_prediction()
st.divider()