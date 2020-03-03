"""import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
(train_data, train_label), (test_data, test_label) = data.load_data(num_words=10000)    #relevant 10000 words
word_index = data.get_word_index()                      #word mapping with default dictionary
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#trimm data bc the revies have different lengths --> Using max 250#
#Reviews with less thanh 250 gtting PAD until 250
#Revies with more than 250 gets trimmed down
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

# this function will return the decoded (human readable) reviews
print(decode_review(test_data[1]))"""


"""import matplotlib.pyplot as plt

xs = []
ys = []
name = "Anna"
State = "CA"
gender = "F"
with open(r'C:\Users\noahb\Desktop\PythonTut\Kursmaterialien\data\names.csv', "r") as file:
	counter = 0
	for line in file:
		line_split = line.strip().split(",")
		if line_split[1] == name and line_split[4] == State and line_split[3] == gender:

			# print(line_split)
			# print(line_split[2])
			# print(line_split[5])
			xs.append(int(line_split[2]))
			#
			ys.append(int(line_split[5]))
		# print(ys)
print(len(xs))
print(len(ys))
#print(xs)
#print(ys)
plt.plot(xs, ys)
plt.show()

# print(line_split)
# print(line.strip().split(","))
# print(line)
# counter = counter + 1
# if counter == 4:
#   break
"""
