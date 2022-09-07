#Importar o opencsv
import cv2

# importar o modelo treinado
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faz a predição de onde esteja a imagem
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #para cada coordenada de onde esteja a face da pessoa
    #ele vai desenhar um retângulo
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

vc.release()
cv2.destroyWindow("preview")

