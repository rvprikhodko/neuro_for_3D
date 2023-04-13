import keras
import numpy as np
from keras.utils import load_img, img_to_array


model = keras.models.load_model('neuro.h5')

img_path = 'data/test/overheating/ov61.jpg'   # input image path
img = load_img(img_path, target_size=(80, 80))
classes = ['clear', 'overheating', 'stringing']

x = img_to_array(img)
x = 255 - x
x /= 255
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
prediction = np.argmax(prediction, axis=1)
print(classes[prediction[0]])   # вывод предсказания
