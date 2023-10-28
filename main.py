import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

# set title
st.title('Disease classification')
# set header
st.header('Drop your eye image')



# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
# load classifier
model = load_model("C:/Users/Marin/Desktop/Semestre 7/SLM-DataCamp/keras_model.h5", compile=False)
# load class names
with open("C:/Users/Marin/Desktop/Semestre 7/SLM-DataCamp/labels.txt", 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()


# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    response_data = classify(image, model, class_names)
    # Display the response data using Streamlit
    st.write("## {}".format(response_data["class"]))
    st.write("### score: {}%".format(int(response_data["confidence_score"] * 1000) / 10))