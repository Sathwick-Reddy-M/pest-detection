import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import keras
import tensorflow as tf
import numpy as np


def get_best_model():
    best_model = keras.models.load_model("./autoencoder/checkpoint/")
    return best_model


st.set_page_config(page_title="Pest Detection")

# YOLO Model Prediction
st.header("Pest Dectection using YOLO")

y_model = YOLO("./data/best.pt")

image_file = st.file_uploader("Upload a image", ["jpg", "png"])

if image_file is not None:
    image = Image.open(image_file)
    results = y_model.predict(source=image, conf=0.5)
    for i in range(len(results)):
        im_bgr = results[i].plot(conf=False)
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        st.image(im_rgb)


# Autoencoder
st.header("Reduce Noise using Autoencoder")

ae_image_file = st.file_uploader("Upload the image to AE", ["jpg", "png"])

if ae_image_file is not None:
    image = Image.open(ae_image_file)
    image = tf.image.resize(image, [304, 304]).numpy() / 255
    st.image(image)
    best_model = get_best_model()
    predict_img = best_model.predict(np.array([image]))
    st.image(predict_img)

# Implement with Noise reduction
st.header("Autoencoder + YOLO")

noise_image_file = st.file_uploader("Upload the image to AE + YOLO", ["jpg", "png"])

if noise_image_file is not None:
    image = Image.open(noise_image_file)
    image = tf.image.resize(image, [304, 304]).numpy() / 255
    st.image(image)
    best_model = get_best_model()
    predict_img = best_model.predict(np.array([image]))
    st.image(predict_img[0])
    y_image = Image.fromarray(np.uint8(predict_img[0] * 255))
    results = y_model.predict(source=y_image, conf=0.5)
    for i in range(len(results)):
        im_bgr = results[i].plot(conf=False)
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        st.image(im_rgb)
