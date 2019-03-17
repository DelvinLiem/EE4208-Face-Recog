import numpy as np
import os
import sys
import cv2
import pandas as pd
# import argparse


def extract_eigenvector(mean_deducted):
   
    cols = np.shape(mean_deducted)[1]
    cov = np.dot(mean_deducted.T , mean_deducted)/(np.shape(mean_deducted)[0] - 1)

    eigvalue, eigenvectors = np.linalg.eigh(cov, UPLO='L')

    sorted_index = np.argsort(eigvalue)
    eigenvectors_sort = np.copy(eigenvectors[:, sorted_index])[:, cols-200-1:cols-1]
    eigenvectors_fin = np.copy(eigenvectors_sort)
    eigenvectors_normalized = np.copy(eigenvectors_sorted)
    for i in range(0, cols):
        eigenvectors_normalized[i] = eigenvectors_normalized[i]/np.linalg.norm(eigenvectors_normalized[i]) * scale

    np.savetxt(EIGVEC, eigenvectors_fin, delimiter=',')
    np.savetxt(EIGVEC_NORM, eigenvectors_normalized,delimiter=',')
    for i in range(199,150,-1):
        cv2.imwrite("pca_eigenface/pca_eigenface_"+str(i)+".jpg", np.reshape(eigenvectors_fin[:,i], (100,100)) )
        cv2.imwrite("pca_eigenface/pca_eigenface_norm_"+str(i)+".jpg",np.reshape(eigenvectors_normalized[:,i],(100,100)) )
    
    if norm:
        return eigenvectors_normalized
    return eigenvectors_fin



def pca_normal(image_matrix):
    
    

    col_size = np.shape(image_matrix)[1]
    faces = np.shape(image_matrix)[0]

    average = [None] * col_size

    #calculate average values of faces, and the matrix deducted from it
    mean_deducted = np.zeros(np.shape(image_matrix))
    for i in range(0, col_size):
        average[i] = np.mean(image_matrix[:,i])
        mean_deducted[:,i] = image_matrix[:,i] - average[i]

    #write out resulting average face values
    ave = np.array(np.reshape(average, (100,100)))
    np.savetxt("./average_face_test.csv", average, delimiter=',')
    cv2.imwrite("average_face.jpg", ave)
    #sys.exit(0)

    #calculate the eigenvectors for the mean deducted matrix
  
    eigenvector = get_eigenvector(mean_deducted, normalize)
    
    #calculate the reduced face matrix and return
    mean_deducted = np.array(mean_deducted)
    reduced = np.zeros((np.shape(mean_deducted)[0], np.shape(eigenvector)[1]))
    for i in range(0,faces):
        reduced[i] = np.dot(mean_deducted[i], eigenvector)

    return reduced
