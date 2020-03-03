import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
#print(data.head())

le = preprocessing.LabelEncoder()
#Die attribute von buying werden in eine Liste gespeichert und in integer zahlen umgewandelt
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
print(buying)

predict = "class"

X = list(zip(buying, maint, door, persons, lug, safety)) #zip into one big list (in tuples)
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=7)     # k=7 --> acc = 95,6%; k=5 --> 94,2%;
model.fit(x_train, y_train)                     #
accuracy = model.score(x_test, y_test)
print(accuracy)
predicted = model.predict(x_test)
#print(predicted)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 7, True)
    print("N: ", n)
#print(x_train, y_test) """