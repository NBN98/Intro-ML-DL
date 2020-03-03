import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle
from matplotlib import style        #style of the grid

data = pd.read_csv("student-mat.csv", sep=";") #Separator um die Werte korrekt zu trennen (da semikolon)

#print(data.head())

data = data[["G1", "G2", "G3", "absences", "freetime", "studytime"]] #nur integer datensätze erstmal nehmen
print(data.head())

predict = "G3"      #G3 predicten, basierend auf den Daten von G1, G2, absences...(G3 = label)
X = np.array(data.drop([predict], 1))   #predict fällt weg
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
best = 0
"""for _ in range(30):

#Splittet die Daten/Attribute in 4 Arrays (x_train von X, y_train von Y; test_size teilt 10% des Datensatzes auf für x_test und y_test )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    linear = linear_model.LinearRegression()    #Traingsmodel Lineare Regression

    linear.fit(x_train, y_train)    #Sucht best fit line mit den Werten aus x_train und y_train
    accurary = linear.score(x_test, y_test)     #Zeigt wie genau das Modell ist
    print(accurary)     #im Schnitt zwischen 70 und 95%
                        #mehr Datensätze haben nur manchmal zu einem besseren Modell geführt
    if accurary > best:
        best = accurary
        with open("studentmodelgrade.pickle", "wb") as f:
            pickle.dump(linear, f)              #safe the model """

pickle_in = open("studentmodelgrade.pickle", "rb")
linear = pickle.load(pickle_in)     #lädt die pickel datei in die linear variable --> Muss nicht mehr trainert werden das Modell

print("Steigung m: \n", linear.coef_) #hier mehrdimensional
print("Veschiebung um t: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    # erste Spalte predictions, zweite Spalte ist ein Array mit den verschiedenen Attributen und dritte Spalte ist das ergebnis y
    print(predictions[x], x_test[x], y_test[x])
p = "absences"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
