""""
audio : MFCC aggr�g� + r�gression logistique (variante s�ance 3)
audio : MFCC + CNN (s�ance 4)
audio : repr�sentations d�j� apprises + couche de sortie (s�ance 4)
image : hog + r�gression logistique (s�ance 3)
image : ResNet (s�ance 4)
"""


#TODO renommer le fichier
import cv2
import numpy as np
def resize(img,output_shape):#used for ResNet
    img_shape=img.shape
    x,y,c=img_shape
    if img_shape==output_shape:
        return img


    img = cv2.resize(img, dsize=output_shape, interpolation=cv2.INTER_CUBIC)

    return img

def batch():
    pass
