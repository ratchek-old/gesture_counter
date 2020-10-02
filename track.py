import cv2
import numpy as np
import argparse

#TODO why are eyes flipped

def detect_eyes(gray_face, classifier):
    #We're doing two things.
    # 1. Making sure that the eyes are above the middle of the face
    # 2. Labeling the left and right eye
    eyes = classifier.detectMultiScale(gray_face, 1.3, 5)
    # get face frame height and width
    width = np.size(gray_face, 1)
    height = np.size(gray_face, 0)
    # Make sure that if you don't find eyes, you don't throw errors
    left_eye, right_eye = None, None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        # get the eye center and determine if it's a left eye or a right eye
        eyecenter = x + w / 2
        if eyecenter < width * 0.5:
            left_eye = (x, y, w, h)
        else:
            right_eye = (x, y, w, h)
    return left_eye, right_eye

# Grab an input file from commandline
ap = argparse.ArgumentParser()
ap.add_argument("file")
args = ap.parse_args()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread(args.file)

#make picture gray
gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)


for (x,y,w,h) in faces:
    # parameters are image, start_point, end_point, color, and thickness
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
     # cut the gray face frame out
    gray_face = gray_picture[y:y+h, x:x+w]
     # cut the face frame out
    face = img[y:y+h, x:x+w]
    le, re = detect_eyes(gray_face, eye_cascade)
    ex, ey, ew, eh = tuple(le)
    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    ex, ey, ew, eh = tuple(re)
    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
# for (ex,ey,ew,eh) in eyes:
#     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)


# window_name, image
# cv2.imshow('my image',img)
cv2.imshow('my face', img)
# displays the image till keypress. cv2.waitKey(1) would display it for 1 ms
cv2.waitKey(0)
cv2.destroyAllWindows()
