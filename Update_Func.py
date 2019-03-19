import numpy as np
import cv2
import sys
import os
import shutil
import argparse
import threading
import time
import pandas as pd
import math
# from sklearn.preprocessing import StandardScaler

main_db = "./database"

def capture_face(data_folder, photo_count, name):
    
    cam = cv2.VideoCapture(1)
    
    os.mkdir(data_folder)
    count = 0
    '''
    print("made path")
    sys.exit(0)
    '''
    timer = 0
    while(count < photo_count):

        #read a frame from the camera in grayscale
        _,frame = cam.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #prepare face detector/cascade
        face_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_frontalface_default.xml')

        #returns coordinate, width and height of each detected faces in frame (x,y,w,h)
        faces_coordinate = face_cascade.detectMultiScale(frame, 1.3, 5)

        face_rect = []

        cv2.namedWindow('Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Inputted Face', cv2.WINDOW_NORMAL)

        #collect actual faces from the frame
        for (x,y,w,h) in faces_coordinate:

            #append actual faces from a frame to a list, via coordinate and width/height adjustment
            face_rect.append(frame[y : y + h, x : x + w])

            #draw rectangles around each detected faces
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #if already at interval,
        if ((timer % 100) == 10) and len(face_rect):

            norm_size = (100,100)
            final_faces = []

            #resize all faces to be 100 x 100
            for face in face_rect:
                face_normalized = None
                if face.shape < norm_size:
                    face_normalized = cv2.resize(face, norm_size, interpolation=cv2.INTER_AREA)
                else:
                    face_normalized = cv2.resize(face, norm_size, interpolation=cv2.INTER_CUBIC)
                final_faces.append(face_normalized)

            #write the face(s) into a jpg file
            #print final_faces
            print (str(count+1)+ "/" + str(photo_count))
            cv2.imwrite( data_folder + '/' + name + '_' + str(count+1)+'.jpg', final_faces[0])
            cv2.imshow('Inputted Face', final_faces[0])
            count +=1 

        cv2.imshow('Feed', frame)
        cv2.waitKey(100)

        timer += 10

def update_database(folder):
    
    #face_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_eye.xml')

    photo_count = None

    while(type(photo_count) != int):
        photo_count = int(input("Photo count => "))

    name = input(" Name: ").lower()

    count = 0

    #create path with folder + name
    dir_path = folder + '/' + name 

    #if path does not exist yet
    if not os.path.exists(dir_path):
        capture_face(dir_path, photo_count, name)


    #if path already exists, then no need to take a picture of user anymore
    else:
        print ("data already exists")



