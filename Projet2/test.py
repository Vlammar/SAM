""""
audio : MFCC aggrege + regression logistique (variante seance 3)
audio : MFCC + CNN (seance 4)
audio : representations deja apprises + couche de sortie (seance 4)
image : hog + regression logistique (seance 3)
image : ResNet (seance 4)
"""


#TODO renommer le fichier
#import cv2
import numpy as np
def resize(img,output_shape):#used for ResNet
    img_shape=img.shape
    x,y,c=img_shape
    if img_shape==output_shape:
        return img


   # img = cv2.resize(img, dsize=output_shape, interpolation=cv2.INTER_CUBIC)

    return img

def batch():
    pass
import sklearn
from sklearn.metrics import roc_curve, auc
print("hello")
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, scores, pos_label=2)
import matplotlib.pyplot as plt
plt.plot(fpr)
plt.plot(tpr)
plt.show()
print(thresholds)
