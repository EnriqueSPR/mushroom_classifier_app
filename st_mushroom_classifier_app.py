# Import necessary libraries

import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import keras
from datetime import date, datetime, timedelta



# Layout
st.set_page_config(
    page_title="mushroom-identificator",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="auto") 

# Page Layout
col1 = st.sidebar
col2, col3 = st.beta_columns((2,1))

# Logo
image = Image.open('logo.png')
st.image(image, width = 700)

# Title
st.markdown("""## **An App to classify mushrooms 🍄**""")

# Description
st.markdown("""

**Description**: This app was built to classify pictures of mushrooms. \n
Two CNNs were trained on a total of 154.000 pictures (size=600x600) from a total of 933 species:
* 1. Based on the [Inception Resnet V2 architecture](https://keras.io/api/applications/inceptionresnetv2/) (Trainable params: 70,857,725).
* 2. Based on the [Xception architecture](https://keras.io/api/applications/xception/) (Trainable params: 39,502,029).

""")

# Mushroom Species
mush = st.checkbox("Note: Click here to check the list of mushrooms.")
if mush:
    list_classes = pd.read_csv("labels_933.csv")
    #list_classes = list(list_classes["Names"])
    st.dataframe(list_classes)
    
st.markdown("---")

# About

expander_bar = st.beta_expander("About", expanded=True)
expander_bar.markdown("""
* **Python libraries used:** pandas, numpy, streamlit, PIL, keras (2.4.0).
* **Data**: Pictures scrapped from the web.
* **Author**: Enrique Alcalde [Find out more about this project](https://enriquespr.github.io/Enrique_portfolio_web/post/project_1/) 🙋🏽‍♂️.
---
""")

col1.subheader("Input")
models_list = ["Inception Resnet V2", "Xception"]
selected_model = col1.selectbox("Select the Model", models_list)

# component to upload images
uploaded_file = col1.file_uploader(
    "Upload a mushroom image to classify", type=["jpg", "jpeg", "png"])

# component for toggling code
show_code = col1.checkbox("Show Code")

# Function to prepare the picture
def prepare_image_Inception_Resnet(im):
    size = (600, 600)
    imag = ImageOps.fit(im, size, Image.ANTIALIAS)
    img_array = np.asarray(imag)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.inception_resnet_v2.preprocess_input(img_array_expanded_dims)

def prepare_image_xception(im):
    size = (600, 600)
    imag = ImageOps.fit(im, size, Image.ANTIALIAS)
    img_array = np.asarray(imag)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.inception_resnet_v2.preprocess_input(img_array_expanded_dims)

def top_5_predictions(array, class_list):
    
    """function to retrieve the top 5 predictions """
    
    inx = array[0].argsort()[-5:][::-1] # getting the indexes of the top 5 predictions in descending order
    
    top_1 = array[0][inx[0]]*100
    top_2 = array[0][inx[1]]*100
    top_3 = array[0][inx[2]]*100
    top_4 = array[0][inx[3]]*100
    top_5 = array[0][inx[4]]*100
    
    class_1 = class_list[inx[0]]
    class_2 = class_list[inx[1]]
    class_3 = class_list[inx[2]]
    class_4 = class_list[inx[3]]
    class_5 = class_list[inx[4]]
    
    return st.code("Top 1 Prediction: With {:4.1f}% probability is a picture of {}.\nTop 2 Prediction: With {:4.1f}% probability is a picture of {}.\nTop 3 Prediction: With {:4.1f}% probability is a picture of {}.\nTop 4 Prediction: With {:4.1f}% probability is a picture of {}.\nTop 5 Prediction: With {:4.1f}% probability is a picture of {}."
                 .format(top_1, class_1, top_2, class_2, top_3, class_3, top_4, class_4, top_5, class_5))

@st.cache
def model_load(path):
    model = keras.models.load_model(path)
    return model

ima = Image.open("Coprinopsis_cineirea.jpeg")
newsize = (250, 250) 
resized = ima.resize(newsize) 
shown_pic = st.image(resized, caption='Upload on the left a "Mistery Mushroom" such as this one')
shown_pic

if uploaded_file:
    ima = Image.open(uploaded_file)
    if selected_model == "Inception Resnet V2":
        newsize = (250, 250) 
        resized = ima.resize(newsize) 
        shown_pic = st.image(resized, caption='Mistery Mushroom')
        shown_pic
        st.write("")
        st.info("Classifying...")

        path = "my_mushroom_model_inception_resnet.h5"
        Inception_Resnet = model_load(path)
        prepared_img = prepare_image_Inception_Resnet(ima)
        predictions_mush = Inception_Resnet.predict(prepared_img)
        list_classes = pd.read_csv("labels_925.csv")
        list_classes = list(list_classes["Names"])
        
        st.balloons()
        st.success("See below the top 5 results and the corresponding probability...")
        results = top_5_predictions(predictions_mush, list_classes)

    if selected_model == "Xception":
        newsize = (250, 250) 
        resized = ima.resize(newsize)
        st.image(resized, caption='Mistery Mushroom')
        st.write("")
        st.info("Classifying...")

        path = "my_mushroom_model_xception_sgd_.h5"
        Xception = model_load(path)
        prepared_img = prepare_image_Inception_Resnet(ima)
        predictions_mush = Xception.predict(prepared_img)
        list_classes = pd.read_csv("labels_933.csv")
        list_classes = list(list_classes["Names"])
       
        st.balloons()
        st.success("See below the top 5 results and the corresponding probability...")
        top_5_predictions(predictions_mush, list_classes)

    
  
