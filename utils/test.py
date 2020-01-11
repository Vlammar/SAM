
""""
audio : MFCC aggrege + regression logistique (variante seance 3)
audio : MFCC + CNN (seance 4)
audio : representations deja apprises + couche de sortie (seance 4)
image : hog + regression logistique (seance 3)
image : ResNet (seance 4)
"""
import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
#import cv2

#TODO renommer le fichier

def resize(img,output_shape):#used for ResNet
    img_shape=img.shape
    x,y,c=img_shape
    if img_shape==output_shape:
        return img
   # img = cv2.resize(img, dsize=output_shape, interpolation=cv2.INTER_CUBIC)
    return img
