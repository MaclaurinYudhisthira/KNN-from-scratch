import numpy as np
from scipy.stats import mode 

class KNN:
    def __init__(self,k=1):
        self.k=k
        self.train_Data=None
        
    def fit(self,X,y):
        x=np.array(X,dtype=np.float64)
        y=np.array(y,dtype=np.float64).reshape(x.shape[0],1)
        self.train_Data=np.hstack((x,y))
        self.n_features=x.shape[1]
        
    def predict(self,X):
        X=np.array(X)
        results=[self._predict(x) for x in X]
        return np.array(results)
    
    def _predict(self,x):
        distances=[]
        for row in self.train_Data:
            dist=self.getDist(row[:self.n_features],x)
            distances.append(dist)
        _idxs=np.argsort(distances)
        neighbors=self.train_Data[_idxs[:self.k]]
        return mode(neighbors[:,-1])[0]
    
    def getDist(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))