import re
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
dataset = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t' , quoting = 3)
clean_sents = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    clean_sents.append(review)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 1500)

x = cv.fit_transform(clean_sents).toarray()
y = dataset.iloc[:, -1].values
bag_of_words = cv.get_feature_names()

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.20, random_state = 6)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test, y_predict)
accuracy_value = (accuracy[0][0] + accuracy[1][1])  / (accuracy[0][0]+accuracy[0][1]+accuracy[1][0]+accuracy[1][1])




testing_clean_sents = []
text = input("enter a review ")
review = re.sub('[^a-zA-Z]', ' ', text)
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if word not in stop_words]
review = ' '.join(review)
for word in bag_of_words:
    ii= review.count(word)
    testing_clean_sents.append(ii)

y_predic = classifier.predict([testing_clean_sents])

if y_predic == 0:
    print("It is a negative review")
else:
    print("it is a positive review")    
        
        

    