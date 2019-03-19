import sys
import cv2
import numpy as np
import os

main_db = "./database"

def read_from_db(database):

    entries = [person for person in os.listdir(database)]

    entries.sort()

    image_list = []
    index =[]

    for count, name in enumerate(entries):

        folder_path = database + '/' + name

        for face in os.listdir(folder_path):
            
            # image index listing
            index.append(face.split('.')[0])

            # read image in grayscale
            image = cv2.imread(folder_path+ '/' + face, 0)
            
            # retrieve exactly 10000 pixels
            image = image.flatten()
            image = np.reshape(image, (1, len(image)))
            image = image[:10000]
            
            # populate the image list
            if len(image_list) < 1:
                image_list = image
            else:
                image_list = np.append(image_list, image, axis=0)
               
    # save all the image indexes 
    np.savetxt("name_index.csv", index, delimiter='  ,  ', fmt = '%s')

    return image_list

print(read_from_db(main_db))

