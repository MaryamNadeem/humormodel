import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score



def create_tfidf_training_data(docs,column):
    y = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    t=vectorizer.transform(column)

    return t
import pandas as pd

d = []
#load data file
dataFile = open('output1.txt', 'rb')
d = pickle.load(dataFile)
#load the saved sgd model
filename = 'partial_fit_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#loading test data
my_csv1 = pd.read_csv('test.csv', encoding="ISO-8859-1")
column1 = my_csv1.body
actual1=my_csv1.humor
actual1=actual1.values
ylabels1=[]
#string required by model converting int labels to str
for l in range(len(actual1)):
    ylabels1.append(str(actual1[l]))
#creating feature vectors
X = create_tfidf_training_data(d,column1)

#loading train data
my_csv2 = pd.read_csv('train.csv', encoding="ISO-8859-1")
column2 = my_csv2.body
actual2=my_csv2.humor
actual2=actual2.values
ylabels2=[]
#string required by model converting int labels to str

for l in range(len(actual2)):
    ylabels2.append(str(actual2[l]))
#creating feature vectors
X2 = create_tfidf_training_data(d,column2)
#incrementally training the model
loaded_model.partial_fit(X2,ylabels2)

#getting results
result = loaded_model.predict(X)

#calculating accuracies
print ("calculating accuracies...")
s = accuracy_score(ylabels1, result)
print ("Accuracy is ",s*100, "%")

