import torchvision.transforms as T
from PIL import Image

preprocessing_func = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor()
    ]
)

def preprocess(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    return preprocessing_func(img).unsqueeze(0)