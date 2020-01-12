from scipy.signal import resample
import numpy as np

def turnkBykNaive(X,k):
	n=X.shape[0]
	p = int(n/k)
	q =  int(n/k)
	return X[::p+1,::q+1,:]

def turnkBykMean(X,k):
	n=X.shape[0]
	res = np.zeros((k,k,3))

	for i in range(k):
		for j in range(k):
			for l in range(3):
				res[i,j,l] = np.mean(X[int(i*n/k):int((i+1)*n/k),int(j*n/k):int((j+1)*n/k),l],axis=None)
	return res/256

def turnkbykClean(X,k):
	return resample(resample(X,3,axis=0),3,axis=1)
