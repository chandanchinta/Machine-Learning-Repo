from nltk.tokenize import word_tokenize
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import re
all_words = []
#importing the data
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#converting words present in each review to arrays
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    all_words.append(word_tokenize(review))

#reading each word from all the reviews
words = []
for i in range(len(all_words)):
    for word in all_words[i]:
        words.append(word)

#removing repeated words
clean_words = []
for word in words:
    if word not in clean_words:
        clean_words.append(word)

#assigning a number to each word and creating sequence of words
count = []
for i in range(0,len(clean_words)):
    count.append(i)
sequence_of_words = dict(zip(clean_words,count))

#saving the sequence of words
import pickle 
with open('sequenceofwords' , 'wb') as fid:
    pickle.dump(sequence_of_words , fid)
#reading the labels into an array
labels =[]
for i in range(0,1000):
    labels.append(dataset.Liked[i])

#converting each review to sequence of words
reverse_mapping=[]
for i in range(0, 1000):
    lop = []
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    for text in review.split():
        if text not in sequence_of_words.keys():
            lop.append('0')
        if text in sequence_of_words.keys():
            lop.append(sequence_of_words[text])
    reverse_mapping.append(lop)   

#deviding the data for training and testing
trainer_data = list(reverse_mapping[:700])
train_labels = [labels[:700]]
tester_data = list(reverse_mapping[700:])
test_labels = [labels[700:]]

train_data = keras.preprocessing.sequence.pad_sequences(trainer_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(tester_data, value = 0 , padding = 'post' , maxlen = 256)

#creating the neural network
vocab_size = 2021
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_length = 256))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

#configuring the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#training the model
model.fit(train_data, train_labels ,epochs = 40, batch_size=6, verbose= 1)

model.summary()

results = model.evaluate(test_data, test_labels)

print(results)

#saving the model
keras_file = "training.h5"
keras.models.save_model(model, keras_file)




