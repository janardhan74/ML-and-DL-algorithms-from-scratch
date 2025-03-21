from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

from decision_tree_classification import DecisionTreeClassifier

X,y = make_classification(n_samples=200,n_features=4,n_informative=3,n_redundant=1,n_classes=3)

classifier = DecisionTreeClassifier(max_depth=20)
classifier.fit(X,y)

preds = classifier.predict(X)

def accuracy(y_pred,y_true):
    acc = (y_pred==y_true).sum() / len(y_pred)
    return acc

print(accuracy(preds,y))

