#%%Importar o opencsv
import cv2
import tensorflow as tf
import numpy as np

#%%
model = tf.keras.models.load_model('model_optimal.h5')

label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

#%%
# importar o modelo treinado
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#define o nome da janela que será utilizada nas imagens
cv2.namedWindow("preview")

#prepara a webcam
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    #esse vc.read() retorna um frame e esse rval que eu não sei o que é
    rval, frame = vc.read()

    frame_resize = cv2.resize(frame, (48, 48))
    frame_resize = frame_resize[:,:,0]
    bau = np.expand_dims(frame_resize,axis = 0)
    print(frame_resize.shape) 
    frame_resize = bau.reshape(1,48,48,1)

    result = model.predict(frame_resize)
    result = list(result[0])


    label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    img_index = result.index(max(result))
    print(label_dict[img_index])
    
    
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faz a predição de onde esteja a imagem
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    frame_resize = cv2.resize(frame, (48, 48))
    frame_resize = frame_resize[:,:,0]
    bau = np.expand_dims(frame_resize,axis = 0)

    frame_resize = bau.reshape(1,48,48,1)

    result = model.predict(frame_resize)
    result = list(result[0])


    label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    img_index = result.index(max(result))
    label_final = label_dict[img_index]

    #para cada coordenada de onde esteja a face da pessoa
    #ele vai desenhar um retângulo
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
      cv2.putText(frame, label_final, (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),  2,cv2.LINE_AA)

vc.release()
cv2.destroyWindow("preview")

