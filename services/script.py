from fastapi import FastAPI
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
from feat.detector import Detector

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='svm',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)



app = FastAPI()

@app.post("/predict_image/")
def predict_image(image):
    
    
    return detector


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8052)