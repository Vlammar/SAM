import sys
import os
sys.path.append('../utils')
from load import *
from msdi_io import *
from KbyK import *
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
#msdi_path = '/home/mrb8/Bureau/M2/SAM/projet_v2/datas/msdi'


class batchLoaderHOG(batchLoader):
#	@Override
	def loadBatch(self):
		batch_size = min(self.i+self.batch_size,self.max_size)
		X = []
		y = []

		for i_batch in range(batch_size):
			if i_batch%10 == 0 :
				os.system('cls' if os.name == 'nt' else "printf '\033c'")
				print(str(i_batch)+"/"+str(self.batch_size))
			entry_idx = self.i + i_batch
			one_entry = self.msdi.loc[entry_idx]

			img = load_img(one_entry, self.path_msdi)
			genre =  get_label(one_entry)
			fd = hog(rgb2gray(img), orientations=4,pixels_per_cell=(199/3, 199/3), visualize=False)	
			y.append(genre)

			X.append(fd)
		self.i += i_batch
		return  np.array(X),y


class batchLoader3x3(batchLoader):
#	@Override
	def loadBatch(self):
		k=3
		batch_size = min(self.i+self.batch_size,self.max_size)
		X = []
		y = []
		for i_batch in range(batch_size):
			if i_batch%10 == 0 :
				os.system('cls' if os.name == 'nt' else "printf '\033c'")
				print(str(i_batch)+"/"+str(self.batch_size))
			entry_idx = self.i + i_batch
			one_entry = self.msdi.loc[entry_idx]

			img = load_img(one_entry, self.path_msdi)
			genre =  get_label(one_entry)
			X.append(np.array([turnkBykNaive(img,k),turnkbykClean(img,k),turnkBykMean(img,k)]).reshape(-1))

			y.append(genre)

		self.i += i_batch
		return  np.array(X),y


if __name__ == '__main__':
	print('Labels:', get_label_list())
	bl3 = batchLoader3x3(3000,path_msdi=msdi_path)
	blH = batchLoaderHOG(3000,path_msdi=msdi_path)
	clf = SVC(C=10,kernel='rbf')

	print('='*10,"3x3",'='*10)
	batch = bl3.loadBatch()
	X_3x3,y = batch[0],batch[1]
	print(cross_val_score(clf,X_3x3,y,cv=5))
	
	print('='*10,"HOG",'='*10)
	batch = blH.loadBatch()
	X_HOG,y = batch[0],batch[1]
	print(cross_val_score(clf,X_HOG,y,cv=5))
	

	print(cross_val_score(clf,np.hstack((X_HOG,X_3x3)),y,cv=5))
	
