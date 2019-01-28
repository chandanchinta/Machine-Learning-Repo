import re
import keras
import pickle

from keras.models import load_model
model = load_model('training.h5')


with open('sequenceofwords' , 'rb') as fid:
    sequence_of_words = pickle.load(fid)

predictor = []
pop = input("enter a review to classify it : ")

def prediction():
    poper = re.sub('[^a-zA-Z]',' ', pop)
    poper = poper.lower()
    poper = poper.split()
    prep = []
    for word in poper:
        if word not in sequence_of_words.keys():
                predictor.append('0')
        if word in sequence_of_words.keys():
                prep.append(sequence_of_words[word])
    prep = keras.preprocessing.sequence.pad_sequences([prep], value = 0 , padding = 'post' , maxlen = 256)
    
      
    
    label = model.predict_classes([prep])
    
    if label[0] == 0 :
        print("It is a negative review")
    if label[0] == 1:
        print("It is a positive review")

prediction()        