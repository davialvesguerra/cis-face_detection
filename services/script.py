from fastapi import FastAPI
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
from feat.detector import Detector
from feat.utils import get_test_data_path
from feat.plotting import imshow
import os

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='svm',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)


# Get the full path
# single_face_img_path = os.path.join(test_data_dir, "single_face.jpg")
single_face_img_path = 'services/fear.jpg'


single_face_prediction = detector.detect_image(single_face_img_path)

result_emotions = single_face_prediction.emotions.max()
emotion_winner = result_emotions[result_emotions == max(result_emotions)]
emotion_winner = emotion_winner.index[0]


app = FastAPI()

@app.get("/predict_image/")
def predict_image():
    return emotion_winner


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8052)