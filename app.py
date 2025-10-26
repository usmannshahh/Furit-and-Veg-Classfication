import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

model = load_model("D:\Furits_vegatables_data\furit\veg_model.keras")
img_hight = 180
img_width = 180
image = 'D:\Furits_vegatables_data\apple.jpg'

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

image_load = tf.keras.utils.load_img(image,target_size=(img_hight,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predcit = model.predict(img_bat)

score = tf.nn.softmax(predcit)

st.image(image)


print('Veg/fruit in the image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score)*100))



