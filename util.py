from keras.models import load_model
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np



def set_background(image_file):
    background_image=Image.open("adn.jpg")
    return background_image


def classify(image,model,class_names):

    # Prétraitement de l'image  
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Créer l'array d'entrée pour le modèle
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prédire la classe
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Renvoyer la prédiction et le score de confiance au format JSON
    response_data = {
        "class": class_name,
        "confidence_score": float(confidence_score)
    }
    return response_data