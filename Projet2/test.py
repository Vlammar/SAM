""""
audio : MFCC aggrégé + régression logistique (variante séance 3)
audio : MFCC + CNN (séance 4)
audio : représentations déjà apprises + couche de sortie (séance 4)
image : hog + régression logistique (séance 3)
image : ResNet (séance 4)
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
