import streamlit as st
import time
import torch
from PIL import Image
import requests
from io import BytesIO
from torchvision.models import resnet18
from models.model_1.preprocessing import preprocess    # здесь мы картинки обрабатываем и вписываем в нужный нам размер


st.markdown(
    '<h1 style="text-align: center;">Модель, предсказывающая погоду 🐛</h1>',
    unsafe_allow_html=True
)
st.write('**Пользователь загружает картинку (или несколько, если используется загрузчик) в модель. Модель дает свое предсказание класса для картинки.**')

uploaded_files = st.file_uploader('Загрузите изображение', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
image_url = st.text_input('Или вставьте ссылку на изображение')

classes = ['dew 💦','fogsmog 💨','frost ❄️','glaze ⛄️','hail 🌨','lightning ⚡️','rain 🌧','rainbow 🌈','rime ❄️','sandstorm 🌪','snow 🌨']
image = None

if uploaded_files:
    image = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        image.append(img)
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f'Не удалось загрузить изображение по ссылке. Ошибка {e}')
else:
    st.write('Выберите способ загрузки изображения!')

@st.cache_resource()
def load_model():
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 11)
    model.load_state_dict(torch.load('models/model_1/resnet_weather_weights.pt', map_location=torch.device('cpu')))
    return model

model = load_model()     # создаем экземпляр класса
model.eval()

def predict(img):
    img = preprocess(img)
    pred = model(img)
    return pred


if image:

    start_time = time.time()

    if isinstance(image, list):
        for img in image:
            st.image(img, caption='Загруженное изображение')
            # сюда вызываем предикт модели
            prediction = predict(img)
            st.write(f'Предсказанный класс:')
            st.markdown(f"""
            <h2 style='text-align: center; color:#3262a8; font-size: 30px; font-weight: bold; padding: 10px; bolder-radius:10px;'>
            {classes[prediction.argmax(axis=1).item()]}
            </h2>
            """, unsafe_allow_html=True)
    else:
        st.image(image, caption='Загруженное изображение')
        # сюда вызываем предикт модели
        prediction = predict(image)
        st.write(f'Предсказанный класс:')
        st.markdown(f"""
        <h2 style='text-align: center; color:#3262a8; font-size: 30px; font-weight: bold; padding: 10px; bolder-radius:10px;'>
        {classes[prediction.argmax(axis=1).item()]}
        </h2>
        """, unsafe_allow_html=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write('Время выполнения предсказания в секундах:')
    st.markdown(f"""
    <h3 style='text-align: center; color:#3262a8; font-size: 30px; font-weight: bold; padding: 5px; bolder-radius:5px;'>
    {elapsed_time:.2f}
    </h3>
    """, unsafe_allow_html=True)

else:
    st.stop()

