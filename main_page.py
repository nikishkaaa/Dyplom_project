import streamlit as st
import base64

# Функция для конвертации изображения в base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def display_home_page():
    image_base64 = image_to_base64("images/churn.png")

    st.markdown("<h1 style='text-align: center;'>Прогнозування відтоку клієнтів банку</h1>", unsafe_allow_html=True)
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style='min-height: 600px; display: flex; align-items: center; justify-content: center; background-color: #f9f9f9; padding: 10px; border-radius: 10px;'>
                <img src="data:image/png;base64,{image_base64}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style='min-height: 600px; display: flex; flex-direction: column; justify-content: center; padding: 10px; border-radius: 10px; font-size:18px; line-height: 1.6;'>
                <h3>🔍 Про застосунок</h3>
                <p>
                    Ця програма прогнозує ймовірність відтоку клієнта (тобто припинення користування послугами банку) 
                    на основі аналізу поведінки та профілю користувача з використанням моделей машинного навчання.
                </p>
                <h3>📊 Основні можливості</h3>
                <ul>
                    <li>Аналіз моделей використання, демографічних даних та історії обслуговування</li>
                    <li>Візуалізація історичних даних у вигляді інтерактивних графіків</li>
                    <li>Прогноз ймовірності відтоку для нового клієнта</li>
                    <li>Допомога компаніям в утриманні клієнтів через таргетовані пропозиції</li>
                </ul>
                <h3>🎯 Для чого це потрібно?</h3>
                <p>
                    Завдяки аналітиці та прогнозам, бізнес може завчасно виявляти клієнтів із високим ризиком 
                    відтоку та вживати ефективних заходів для їхнього утримання.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
st.divider()