import numpy as np

class PCA():
    def __init__(self,n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self,X):
        # find mean of x
        self.mean = np.mean(X,axis=0)
        # subract mean from data
        X = X - self.mean
        # find covariance matrix
        cov_matrix = np.cov(X,rowvar=False)
        # find eigen values , eigen vectors for covariance matrix
        eigen_values,eigne_vectors = np.linalg.eig(cov_matrix)
        # v[:,i] -> eigen vector
        eigen_values = eigen_values.T
        # sort the eigen value in decreasing order
        idxs = np.argsort(eigen_values)
        eigen_values = eigen_values[idxs]
        eigne_vectors = eigne_vectors[idxs]
        # take n_components eigen vectors
        self.components = eigne_vectors[0:self.n_components]
        
    def transfrom(self,X):
        # subract mean from X
        X = X - self.mean
        # transfrom n features into n_componenst by dot product
        # (mxn).(n_componentsxn) -> invalid
        # (mxn).(nXn_components) -> valid ->.T
        return np.dot(X,self.components.T)
    


        
        