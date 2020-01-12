from msdi_io import *

class batchLoader():
	def __init__(self,batch_size,path_msdi,max_size=30712):
		self.i = 0
		self.batch_size= batch_size
		self.path_msdi = path_msdi
		self.max_size = max_size
		self.msdi = get_msdi_dataframe(msdi_path)


	def load(self,batch_nb):
		batch_size = min(self.i+self.batch_size,self.max_size)

		X = []
		y = []
        #on charge directement le batch
		entry_idx=np.arange(batch_nb*batch_size,batch_size*(1+batch_nb))
		one_entry = self.msdi.loc[entry_idx]
		#print(one_entry)

		for index,row in one_entry.iterrows():

			mfcc = load_mfcc(row, self.path_msdi)[:200]
			genre =  get_label(row)
			if len(mfcc) < 200:
				print("donnee trop courte (refusee) de type :",genre)
				continue
			img = load_img(row, self.path_msdi)
			deep_features = load_deep_audio_features(row, self.path_msdi)
			y.append(genre)
			data = np.hstack([mfcc.reshape(-1),img.reshape(-1),deep_features.reshape(-1)])

			X.append(data)
		return  np.array(X),y
	def loadBatch(self):
		batch_size = min(self.i+self.batch_size,self.max_size)
		X = []
		y = []

		for i_batch in range(batch_size):
			entry_idx = self.i + i_batch
			one_entry = self.msdi.loc[entry_idx]

			mfcc = load_mfcc(one_entry, self.path_msdi)[:200]
			genre =  get_label(one_entry)
			if len(mfcc) < 200:
				print("donnee trop courte (refusee) de type :",genre)
				continue
			img = load_img(one_entry, self.path_msdi)
			deep_features = load_deep_audio_features(one_entry, self.path_msdi)

			y.append(genre)
			data = np.hstack([mfcc.reshape(-1),img.reshape(-1),deep_features.reshape(-1)])

			X.append(data)
		self.i += i_batch
		return  np.array(X),y

if __name__ == '__main__':
	print('Labels:', get_label_list())
	bl = batchLoader(100,path_msdi=msdi_path)
	for i in range(10):
		X,y=bl.load(i)
		print(X.shape,len(y))
		#print(X[1])
"""	print('Labels:', get_label_list())
	bl = batchLoader(100,path_msdi=msdi_path)
	for i in range(10):
		for batch in bl.loadBatch():
			X,y = batch[0],batch[1]
			print()
			print(X[1])
"""
