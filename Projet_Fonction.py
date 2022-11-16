from skimage.feature import graycomatrix, graycoprops # GLCM
from mahotas.features import haralick
from BiT import bio_taxo # bitdescpy
import numpy as np

def fctGLCM(file):
    feature_vector = []
    co_matrix = graycomatrix(file, [1], [0], 
                             levels=None, symmetric=True, normed=True)
    diss = graycoprops(co_matrix, 'dissimilarity')[0,0]
    feature_vector.append(diss)
    cont = graycoprops(co_matrix, 'contrast')[0,0]
    feature_vector.append(cont)
    ener = graycoprops(co_matrix, 'energy')[0,0]
    feature_vector.append(ener)
    homo = graycoprops(co_matrix, 'homogeneity')[0,0]
    feature_vector.append(homo)
    corr = graycoprops(co_matrix, 'correlation')[0,0]
    feature_vector.append(corr)
    asm = graycoprops(co_matrix, 'ASM')[0,0] # Angular Second Moment
    feature_vector.append(asm)
    return feature_vector # List of 6 featutes

def fctGLCM_2(file):
    feature_vector = []
    distance = 1
    angles = [0, np.pi/2, np.pi/4]
    for angle in angles:
        co_matrix = graycomatrix(file, [distance], [angle], 
                                levels=None, symmetric=True, normed=True)
        diss = graycoprops(co_matrix, 'dissimilarity')[0,0]
        feature_vector.append(diss)
        cont = graycoprops(co_matrix, 'contrast')[0,0]
        feature_vector.append(cont)
        ener = graycoprops(co_matrix, 'energy')[0,0]
        feature_vector.append(ener)
        homo = graycoprops(co_matrix, 'homogeneity')[0,0]
        feature_vector.append(homo)
        corr = graycoprops(co_matrix, 'correlation')[0,0]
        feature_vector.append(corr)
        asm = graycoprops(co_matrix, 'ASM')[0,0] # Angular Second Moment
        feature_vector.append(asm)
    return feature_vector # List of 6 featutes

def fctHaralick(file):
    return haralick(file).mean(0).tolist() # List of 13 featutes
    
def fctBiT(file):
    return bio_taxo(file) # List of 14 featutes
