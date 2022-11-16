#%%Importar o opencsv
import cv2
import tensorflow as tf
import numpy as np

from utils import reshape_image_to_emotions_model

#%%
#importa o modelo de detecção de emoções
detect_emotions_model = tf.keras.models.load_model('model_optimal.h5')

#define o rótulo das emoções utilizadas pelo modelo
emotions_labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# importar o modelo para detecção de faces
detect_face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#define o nome da janela que será utilizada nas imagens
cv2.namedWindow("preview")

#prepara a webcam
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    #esse vc.read() retorna um frame e esse rval que eu não sei o que é
    rval, frame = vc.read()
        
else:
    rval = False

while rval:
    #abre a imagem da webcam
    cv2.imshow("preview", frame)

    rval, frame = vc.read()
    #isso faz a webcam desligar
    key = cv2.waitKey(30)

    if key == 27: # exit on ESC
        break

    #cria uma versão da imagem em tons cinzas
    image_2D = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faz a predição de onde esteja a imagem
    faces = detect_face_model.detectMultiScale(image_2D, 1.1, 4)
    
    image_resized = reshape_image_to_emotions_model(img_2D=image_2D, format_2D=(48, 48))

    emotion_result = detect_emotions_model.predict(image_resized)
    emotion_result = list(emotion_result[0])

    #pegando a maior probabilidade e encontrando qual o índice da mesma
    img_index = emotion_result.index(max(emotion_result))   
    label_final = emotions_labels[img_index]

    #para cada coordenada de onde esteja a face da pessoa
    #ele vai desenhar um retângulo
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
      cv2.putText(frame, label_final, (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),  2,cv2.LINE_AA)

vc.release()
cv2.destroyWindow("preview")

