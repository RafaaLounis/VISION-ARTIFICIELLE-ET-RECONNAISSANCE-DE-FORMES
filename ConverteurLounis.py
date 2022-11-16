# Libraries
from os import listdir
from typing import List
from Projet_Fonction import fctBiT, fctGLCM, fctGLCM_2, fctHaralick
import cv2
import pandas as pd

def main():
    files_path = 'GLAUCOMA_ACRIMA/'
    files_dir: List[str] = listdir(files_path)
    #print(files_dir)
    features = list()
    for GLAUCOMA_ACRIMA in files_dir:
        if GLAUCOMA_ACRIMA == files_dir[0]: # files_dir[0] => 'CT_COVID'
            label = 1 # classe CT_COVID
            #print('Positive\n----------\n')
            for imageName in listdir(files_path + GLAUCOMA_ACRIMA + '/'):
                imgPath = files_path + GLAUCOMA_ACRIMA + '/' + imageName
                #print(imgPath, label)
                img = cv2.imread(imgPath, 0)                
                feat = fctGLCM(img) # Extraire les caractreristiques
                feat_classe = feat + [label]# Concatenation des caracteristiques avec la classe de l`image
                features.append(feat_classe)
        elif GLAUCOMA_ACRIMA == files_dir[1]: # files_dir[1] => 'CT_NonCOVID'
            label = 0 # classe CT_NonCOVID
            #print('Negative\n----------\n')
            for imageName in listdir(files_path + GLAUCOMA_ACRIMA + '/'):
                imgPath = files_path + GLAUCOMA_ACRIMA + '/' + imageName
                #print(imgPath, label)
                img = cv2.imread(imgPath, 0)                
                feat = fctGLCM(img) # Extraire les caractreristiques
                feat_classe = feat + [label]# Concatenation des caracteristiques avec la classe de l`image
                features.append(feat_classe)
    print('Total # Images: ', len(features))
    # Convert features list to dataframe
    dataframe = pd.DataFrame.from_records(data=features)
    print('Shape: ', dataframe.shape)
    # Convert Dataframe to csv
    dataframe.to_csv('GLAUCOMA_ACRIMA.csv', sep=',', index=False)
if __name__=='__main__':
    main()
