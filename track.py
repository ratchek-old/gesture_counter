import cv2
import numpy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#make picture gray
gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)

for (x,y,w,h) in faces:
    # parameters are image, start_point, end_point, color, and thickness
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

# window_name, image
cv2.imshow('my image',img)
# displays the image till keypress. cv2.waitKey(1) would display it for 1 ms
cv2.waitKey(0)
cv2.destroyAllWindows()
