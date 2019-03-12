import numpy as np
import cv2
import sys

face_cascade = cv2.CascadeClassifier(r"C:\Users\dlvnl\Desktop\EE4208\CascadeClassifiers\haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier(r'C:\Users\dlvnl\Desktop\EE4208\CascadeClassifiers\haarcascade_eye')


#detecting face from video stream

vid = cv2.VideoCapture(1)

for i in range (5):

    #capture frame by frame
    _, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cv2.namedWindow('Feed', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Inputted Face', cv2.WINDOW_NORMAL)

    face_rect = []

    for (x,y,w,h) in faces:

        face_rect.append(frame[y : y + h, x : x + w])

        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# #release everything when done



