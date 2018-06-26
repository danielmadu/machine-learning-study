import numpy as np
import cv2

haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
haar_eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

captura = cv2.VideoCapture(0)

s_img = cv2.imread("grin.png")


# lower = np.array([255,0,255])  #-- Lower range --
# upper = np.array([255,0,255])  #-- Upper range --
# mask = cv2.inRange(grin, lower, upper)

# s_img = cv2.bitwise_and(s_img, s_img, mask= mask)  #-- Contains pixels having the gray color--

while(1):
    ret, frame = captura.read()
    frame = cv2.flip(frame, 1)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		# grin = cv2.resize(s_img, (w,h), interpolation = cv2.INTER_AREA)
		# frame[y:y + grin.shape[0], x:x + grin.shape[1]] = grin

    eyes = haar_eye_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("Video", frame)
   
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

captura.release()
cv2.destroyAllWindows()


