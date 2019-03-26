import numpy as np
import cv2
import sys
import csv
import os

face_cascade = cv2.CascadeClassifier('Haar_Cascades/haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

name = 'kevin'

crop_size = (100,100)

count = 5
save2 = np.empty([5,10000])

if not os.path.exists('database/'+name):
    os.makedirs('database/'+name)

#print save2
while(count):

    #capture frame by frame
    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
    	print faces
        
        save = []
        save1 = []

        roi_gray = gray[y:y+h, x:x+w]

        if roi_gray.shape < crop_size:
            save = cv2.resize(roi_gray, crop_size, interpolation=cv2.INTER_AREA)
        else:
            save = cv2.resize(roi_gray, crop_size, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('gray', save)
        cv2.imwrite('database/'+name+'/'+name+'_'+str(count)+'.jpg',save)

        save1 = save.flatten()
        #save1 = np.append(save1,count)
        save2[count-1] = save1

        #print save2
        
        
        count-=1
        
    np.savetxt(name+".csv", save2, fmt='%i',delimiter=",")
    k = cv2.waitKey(0)
    if (k & 0xFF == ord('q')) or (k & 0xFF == 27):
    	cv2.destroyAllWindows()
    	break

    



vid.release()
