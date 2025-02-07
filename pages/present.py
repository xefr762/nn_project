import streamlit as st

st.title('Отчет о проделанной работе 😎👌')

tab1, tab2 = st.tabs(['Модель 1', 'Модель 2'])
сol1, col2, col3 = st.columns(3)

with tab1:
    st.header('Метрики и графики для модели 1 👀')
    st.write('Начнем с графика LOSS и ACCURACY для первой модели 😏')
    st.image('images/model_1_4.jpg', caption='Loss + accuracy', width=700)
    st.write('Здесь у нас время обучения **АЖ 30 ЭПОХ** на родненькой 1060....Мне ее даже немного жаль.')
    st.image('images/model_1_3.jpg', caption='Время обучения модели')
    st.write('Выведем основные метрики для оценки работы модели')
    st.image('images/model_1_1.jpg', caption='Метрики модели')
    st.markdown(
    '<h3 style="text-align: center;">Метрика F1 для первой модели: 0.856611915458333</h3>',
    unsafe_allow_html=True
)
    st.write('Последний график - матрица корреляции для первой модели')
    st.image('images/model_1_2.jpg', caption='Матрица корреляции')
    st.markdown(
    '<h3 style="text-align: center;">Какой вывод можем сделать? Первая модель отработала неплохо. Нам не удалось пробить потолок в 90%, но мы почти к этому приблизились...</h3>',
    unsafe_allow_html=True
)

with tab2:
    st.header('Метрики и графики для модели 2 👀')
    st.write('Начнем с графика LOSS и ACCURACY для второй модели 😐')
    st.image('images/model_2_3.png', caption='Loss + accuracy', width=700)
    st.write('Здесь тоже обучали на родненькой 1060, но решили сделать эпох поменьше...')
    st.image('images/model_2_1.jpg', caption='Время обучения модели')
    st.write('А вот график общих метрик мы построить не успели... Поэтому **шакальная** картинка с рекламой')
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRM6kOUz_TCd8HCgV5MYBdumjo_7pcKQ_tGLw&s', caption='Метрики модели', width=700)
    st.markdown(
    '<h3 style="text-align: center;">Метрика F1 для второй модели: 0.9027773302694877</h3>',
    unsafe_allow_html=True
)
    st.write('Последний график - матрица корреляции для второй модели')
    st.image('images/model_2_2.jpg', caption='Матрица корреляции')
    st.write('Что здесь вообще происходит?...')
    st.markdown(
    '<h3 style="text-align: center;">Какой вывод можем сделать? Со второй моделью были некоторые проблемы - птиц оказалось больше, чем природных явлений... Поэтому точность меньше.</h3>',
    unsafe_allow_html=True
)