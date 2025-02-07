import streamlit as st

st.title('Добро пожаловать на главную страницу проекта, мастер 🧙‍♂️')

st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            test-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    '<h3 style="text-align: center;">Логотип нашей команды!</h3>',
    unsafe_allow_html=True
)
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image('images/team.jpg', width=300)
st.write('Да, это горящая видеокарта... Прямо как наши...')

st.markdown("""
## Данный проект был создан для демонстрации того, чему мы научились за эту неделю!

**Авторы:** [Илья Крючков](https://github.com/xefr762), [Алина Зарницына](https://github.com/RenaTheDv), [Анатолий Яковлев](https://github.com/cdxxi)

**Описание:**
- **Главная страница**: Общая информация и навигация 🌠
- **Первая страница**: Модель, которая принимает от пользователя картинку и выдает предсказанный класс 🎰
- **Вторая страница**: Модель, которая тоже принимает от пользователя картинку и выдает предсказанный класс (удивительно, да?) 🤔
- **Третья страница**: Метрики моделей, время их обучения, описание проблем 🥂


Переключайтесь между страницами через левый сайдбар! 
""")