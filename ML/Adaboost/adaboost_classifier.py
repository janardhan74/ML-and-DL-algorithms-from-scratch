import numpy as np

class Decisionstump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
    
    def predict(self,X):
        n_samples = X.shape[0]
        
        X_col = X[:,self.feature_idx]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X_col>self.threshold]=-1
        else:
            predictions[X_col<self.threshold]=-1
            
        return predictions
            
        
class AdaBoostClassifier:
    def __init__(self,n_clfs):
        self.n_clfs = n_clfs
        self.clfs = None
        
    def fit(self,X,y):
        
        self.clfs = []
        n_samples,n_features = X.shape
        w = np.full(n_samples,1/n_samples)
        

        for i in range(self.n_clfs):
            
            clf = Decisionstump()
            min_error = float('inf')

            for feat_idx in range(n_features):
                X_column = X[:,feat_idx]
                thresholds = np.unique(X_column)
                
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column>threshold] = -1
                    
                    missclassified = w[y!=predictions]
                    error = np.sum(missclassified)

                    if error > 0.5:
                        error = 1-error
                        p = -1
                        
                    if error < min_error:
                        min_error = error
                        clf.feature_idx = feat_idx
                        clf.polarity = p
                        clf.threshold = threshold
                        
                        
            EPS = 1e-10
            
            clf.alpha = 0.5 * np.log((1-min_error)/(min_error+EPS))
            
            # update weight
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
                    
            self.clfs.append(clf)
            
    def predict(self,X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds,axis=0)

        y_pred = np.sign(clf_preds)
        return y_pred
            
