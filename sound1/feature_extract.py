from os import listdir
import numpy as np
import librosa
from librosa.feature import mfcc,chroma_stft

import pickle
from scipy.io import wavfile  # Pour lire et écrire des fichiers sons


def getMFCC(X):
	return np.array(mfcc(y=X,sr=44100,n_mfcc=1)).reshape(-1,1)

def getChroma(X):
	return chroma_stft(y=X, sr=44100)

def getDelta(mfcc_freqs):
#	mfcc_freqs = mfcc_freqs
	res = []
	for index in range(len(mfcc_freqs)-1):
		res.append(mfcc_freqs[index+1] - mfcc_freqs[index])
	return np.array(res)


#returns MCC, ΔMCC, ΔΔMCC
def getFeatures(X):
	mfcc = getMFCC(X).reshape(-1,1)
	Dmfcc = getDelta(mfcc).reshape(-1,1)
	DDmfcc = getDelta(Dmfcc).reshape(-1,1)

	chromas = getChroma(X).reshape(-1,12)
	Dchromas = getDelta(chromas).reshape(-1,12)
	DDchromas = getDelta(Dchromas).reshape(-1,12)
	
	return mfcc[2:],Dmfcc[1:],DDmfcc,chromas[2:],Dchromas[1:],DDchromas

def getDatas(sound_path):
	features = []
	y = []
	for composer in ['chopin','beethoven','liszt','mozart']:
		for f in listdir(sound_path +'/'+ composer):
			path = sound_path +'/'+ composer+'/'+f
			x = np.array(librosa.load(path,dtype=float)[0][:500000])
			features.append(np.hstack(getFeatures(x)))
			y.append(composer)
	return np.array(features), np.array(y)


def makeDummy(composers):
	composers[composers=='beethoven']=1
	composers[composers=='chopin']=2
	composers[composers=='liszt']=3
	composers[composers=='mozart']=4
	return composers

def computeAndSaveDatas ():
	features,y = getDatas('/media/mrb8/Verbatim/shared/maps_composers_audio/maps_composers')
	print(features.shape)
	np.save("feats",features)
	np.save("composers",y)

	return features,makeDummy(y)

def loadDatas():
	features = np.load("feats.npy",allow_pickle=True)
	composers = np.load("composers.npy",allow_pickle=True)
	return features, makeDummy(composers)

