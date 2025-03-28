from sklearn import datasets
from sklearn.model_selection import train_test_split
from adaboost_classifier import AdaBoostClassifier

X,y=datasets.load_breast_cancer(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y)


abc = AdaBoostClassifier(n_clfs=10)

abc.fit(X_train,y_train)

preds = abc.predict(X_test)

def accuracy(y_true,y_pred):
    acc = (y_true==y_pred).sum() / len(y_true)
    return acc

print(accuracy(preds,y_test))


