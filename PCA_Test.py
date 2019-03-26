import numpy as np
from scipy.linalg import eigh
import os
import sys
import cv2
import pandas as pd

from read_from_db import *

main_db = "./database"

def extract_eigenvector(mean_deducted):
   
    cols = np.shape(mean_deducted)[1]
    cov = np.dot(mean_deducted.T , mean_deducted)/(np.shape(mean_deducted)[0] - 1)

    print ("done cov")
    eigvalue, eigenvectors = np.linalg.eigh(cov, UPLO='L')
    print ("done eigen")

    sorted_index = np.argsort(eigvalue)
    eigenvectors_sort = np.copy(eigenvectors[:, sorted_index])[:, cols-200-1:cols-1]
    eigenvectors_fin = np.copy(eigenvectors_sort)
    # eigenvectors_normalized = np.copy(eigenvectors_sort)
    # for i in range(0, cols):
    #     eigenvectors_normalized[i] = eigenvectors_normalized[i]/np.linalg.norm(eigenvectors_normalized[i]) * scale

    print ('done normalizing')
    np.savetxt(EIGVEC, eigenvectors_fin, delimiter=',')
    # np.savetxt(EIGVEC_NORM, eigenvectors_normalized,delimiter=',')
    for i in range(199,150,-1):
        cv2.imwrite("pca_eigenface/pca_eigenface_"+str(i)+".jpg", np.reshape(eigenvectors_fin[:,i], (100,100)) )
        # cv2.imwrite("pca_eigenface/pca_eigenface_norm_"+str(i)+".jpg",np.reshape(eigenvectors_normalized[:,i],(100,100)) )
    
    # if norm:
    #     return eigenvectors_normalized

    return eigenvectors_fin

def pca_normal(image_matrix):
    
    col_size = np.shape(image_matrix)[1]
    faces = np.shape(image_matrix)[0]

    average = [None] * col_size

    #calculate average values of faces, and the matrix deducted from it
    mean_deducted = np.zeros(np.shape(image_matrix))

    # mean-shift all the values
    for i in range(0, col_size):
        average[i] = np.mean(image_matrix[:,i])
        mean_deducted[:,i] = image_matrix[:,i] - average[i]
        print ("done mean deducting" + str(i+1)+ '/' + str(col_size) )


    #write out resulting average face values
    ave = np.array(np.reshape(average, (100,100)))
    np.savetxt("./average_face_test.csv", average, delimiter=',')
    cv2.imwrite("average_face.jpg", ave)
    

    #calculate the eigenvectors for the mean deducted matrix
    eigenvector = extract_eigenvector(mean_deducted)
    
    #calculate the reduced face matrix and return
    mean_deducted = np.array(mean_deducted)
    reduced = np.zeros((np.shape(mean_deducted)[0], np.shape(eigenvector)[1]))
    print ("done reducing: ")
    for i in range(0,faces):
        reduced[i] = np.dot(mean_deducted[i], eigenvector)
        print (str(i)+'/'+str(faces))

    return reduced


image_table = read_from_db(main_db)
print ("Dimension of image table: ", np.shape(image_table), "\n", image_table)
# np.savetxt("./image_table.csv", image_table, fmt='%i', delimiter=',')
result = None

test_pca = pca_normal(image_table)