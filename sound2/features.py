import sys
sys.path.append('../utils')
from load import *
from msdi_io import *

def getDelta(mfcc_freqs):
	res = []
	for index in range(len(mfcc_freqs)-1):
		res.append(mfcc_freqs[index+1] - mfcc_freqs[index])
	return np.array(res)


class batchLoaderMFCCOnly(batchLoader):
#	@Override
	def loadBatch(self):
		batch_size = min(self.i+self.batch_size,self.max_size)
		X = []
		y = []

		for i_batch in range(batch_size):
			entry_idx = self.i + i_batch
			one_entry = self.msdi.loc[entry_idx]

			mfcc = load_mfcc(one_entry, self.path_msdi)[:202]
			genre =  get_label(one_entry)
			if len(mfcc) < 200:
				print("donnee trop courte (refusee) de type :",genre)
				continue
			y.append(genre)
			Dmfcc = getDelta(mfcc)
			DDmfcc = getDelta(Dmfcc)
			data = np.hstack([mfcc[2:],Dmfcc[1:],DDmfcc])

			X.append(data)
		self.i += i_batch
		return  np.array(X),y

if __name__ == '__main__':
	print('Labels:', get_label_list())
	bl = batchLoaderMFCCOnly(100,path_msdi=msdi_path)
	for i in range(10):
		for batch in bl.loadBatch():
			X,y = batch[0],batch[1]
			print(X[1])
