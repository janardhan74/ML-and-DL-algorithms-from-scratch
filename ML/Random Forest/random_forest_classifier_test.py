from sklearn import datasets
from sklearn.model_selection import train_test_split
from random_forest_classifier import RandomForestClassifier

X,y = datasets.load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y)

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

preds =  rfc.predict(X_test)

def accuracy(y_true,y_pred):
    accuracy = (y_true==y_pred).sum() / len(y_true)
    return accuracy

print(accuracy(y_test,preds))


