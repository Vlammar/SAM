from load import *
bl = batchLoader(100,path_msdi=msdi_path)
for i in range(10):
    X,y=bl.load(i)
