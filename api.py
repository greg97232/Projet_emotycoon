import uvicorn
from fastapi import FastAPI , UploadFile
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import io
from PIL import Image
import cv2
import base64

app = FastAPI ()

@app.get("/")
async def greet():
    # Renvoie un message de bienvenue simple
    return {"message": "Bonjour"}

# Charger la structure du modèle à partir d'un fichier JSON
emotion_model = model_from_json(open("youssef.json", "r").read())

# Charger les poids du modèle
emotion_model.load_weights('youssef.h5')

# Dictionnaire pour mapper les indices aux émotions
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}


# Fonction pour le prétraitement de l'image
def preprocess(image_stream):
    # Conversion d'une image PIL en image OpenCV (tableau numpy)
    frame = np.array(image_stream)

    # Obtenir les dimensions de l'image
    height, width, _ = frame.shape

    # Conversion de l'image en format BGR (utilisé par OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_copy = frame.copy()
    
    # Détection des visages avec CascadeClassifier
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Conversion de l'image en nuances de gris pour la détection des visages
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)
    
    # Initialisation des listes pour stocker les résultats
    face_list = []
    pred_list = []
    full_pred_list = []
    roi_coordinates = [] # Coordonnées des visages détectés
    
    # Traiter chaque visage détecté
    for (x, y, w, h) in num_faces:
        # Convertir la ROI (Region Of Interest) en nuances de gris
        roi_gray_frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x,y), (x+w, y+h+10), (255,255,0), 2)
        
        # Préparer l'image pour la prédiction (redimensionnement et normalisation)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)), -1), 0)

        # Prédiction de l'émotion
        emotion_prediction = emotion_model.predict(cropped_img)
        
        # Récupérer l'émotion avec la plus grande probabilité
        maxindex = int(np.argmax(emotion_prediction))
        
        # Afficher l'émotion prédite
        cv2.putText(frame, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        # Stocker les coordonnées de la ROI
        roi_coordinates.append((x, y, x + w, y + h))
        
        # Ajouter le visage et l'émotion prédite aux listes
        pred_list.append(maxindex)
        full_pred_list.append(emotion_prediction)
        
    # Extraire les visages de l'image originale
    for (x, y, x1, y1) in roi_coordinates:
        face_list.append(frame_copy[y:y1, x:x1])
        
    # Redimensionner l'image avec les rectangles et les textes
    full_img = cv2.resize(frame,(width, height),interpolation = cv2.INTER_CUBIC)
    full_img = np.expand_dims(full_img, 0)  # Ajout des dimensions de batch et de canal
    
    return face_list, pred_list, full_img, full_pred_list

@app.post("/predect")
async def predect(file:UploadFile):
    # Lire l'image envoyée par l'utilisateur
    image_data = await file.read()
    image_stream = Image.open(io.BytesIO(image_data))
    image_stream.seek(0)

    # Prétraitement de l'image avec la fonction preprocess()
    img_processed = preprocess(image_stream)
    
    # Récupérer les prédictions
    predictions = img_processed[1]
    
    img_base64_list = []
    
    # Convertir les images en base64 pour les envoyer au client
    for i in range(len(img_processed[0])):
        _, img_encoded = cv2.imencode('.jpg', img_processed[0][i])
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        img_base64_list.append(img_base64)
    
    # Convertir les images en base64 pour les envoyer au client
    _, full_img_encoded = cv2.imencode('.jpg', img_processed[2][0])
    full_img_base64 = base64.b64encode(full_img_encoded.tobytes()).decode('utf-8')
    
    pred = [None] * len(predictions)
    # Convertir les prédictions en émotions
    for i in range(len(predictions)):
        pred[i] = emotion_dict[predictions[i]]
    
    full_pred = img_processed[3]

    return {'prediction':pred, 'image': img_base64_list, 'full_image': full_img_base64, 'full_prediction': [prediction.tolist() for prediction in full_pred]}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)