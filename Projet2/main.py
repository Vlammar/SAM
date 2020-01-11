from msdi_io import *
_msdi_path = '/home/mrb8/Bureau/M2/SAM/projet_v2/Projet2/datas/msdi'


class batchLoader():

	def __init__(self,batch_size,path_msdi,max_size=30712):
		self.i = 0
		self.batch_size= batch_size
		self.path_msdi = path_msdi
		self.max_size = max_size
		self.msdi = get_msdi_dataframe(_msdi_path)

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
			data = np.hstack((mfcc.reshape(-1),img.reshape(-1),deep_features.reshape(-1)))

			X.append(data)
		self.i += i_batch
		return  np.array(X),y
# Exemple d'utilisation

print('Labels:', get_label_list())
bl = batchLoader(100,path_msdi=_msdi_path)
for i in range(10):
	for batch in bl.loadBatch():
		X,y = batch[0],batch[1]
		print(X[1])
