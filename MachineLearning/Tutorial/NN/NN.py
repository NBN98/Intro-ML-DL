import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist             #load data from a default dataset

(train_images, train_labels), (test_images, test_labels) = data.load_data()     #SPlit data in training and test data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#shrunk down the data
train_images = train_images/255.0
test_images = test_images/255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),          #flattern the data
    keras.layers.Dense(128, activation="relu"),         #dense means every neuron is connected with every other neuron
    keras.layers.Dense(10, activation="softmax")        #probality for every output neuron
    ])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)
#plt.imshow(train_images[1], cmap=plt.cm.binary)
#plt.show()
test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Accuracy: ", test_acc)
prediction = model.predict(test_images)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])                #sucht den wert mit der höchsten probability und mappt den wert
                                                                        #auf die class_names liste
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
#print(class_names[np.argmax(prediction[0])])
#print(train_labels[9])
