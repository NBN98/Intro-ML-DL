import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()        #loading data
#print(cancer.feature_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
print(x_train, y_train)
classes = ["malignant", "benign"]

clf = svm.SVC(kernel="linear", C=1)                 #SVC Support Vector Classification
                                                    # C=1 Soft Margin = 1
                                                    #es gibt viele kernel parameter, hier rumspielen (bei poly sehr großer Rechenaufwand
clf.fit(x_train, y_train)       # Training des Modells

y_pred = clf.predict(x_test)    #basierend auf den Trainingswerten, soll er das Ergebnis predicten

accuracy = metrics.accuracy_score(y_test, y_pred)       #vergleicht beide y werte
print(accuracy)