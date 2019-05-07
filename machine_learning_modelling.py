##necessary imports
from utils import spacy_cleaner
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt 
import seaborn as sns


##Loading the dataset..
train = pd.read_csv("sentiment_analysis/train.csv")
test  = pd.read_csv("sentiment_analysis/test.csv") 

##showing the head..
##train.head(10)
##test.head(10)

##shape  of the dataset..
print("Shape of TRAIN {}:".format(train.shape))
print("Shape of TEST {}:".format(test.shape))

##print the name of columns...
col_names = train.columns
print(col_names)

##cleaning the tweets...
##cleaning
train['clean_text'] = [spacy_cleaner(t) for t in train.tweet]
test['clean_text'] = [spacy_cleaner(t) for t in test.tweet]

##BENCHMARK MODEL..
'''This benchmarking is done with ZeroR rule
'''
##fraction of positive classes
benchmark_score = train.loc[(train['label'] == 1)].shape[0]/train.shape[0]
print("the benchmarking accuracy is {}%".format(benchmark_score*100))


def cross_validation(splits,X,y,pipeline,average_method):
	'''
	This function is used to set the machine learning pipleline...
	arguments..
	splits: number of splits you want in the dataset
	X:--Text data 
	y:--label
	pipeline:--name of the pipeline
	average_method:--metric
	'''
    kfold = StratifiedKFold(n_splits = splits,shuffle = True,random_state = 42)
    acc = []
    precision = []
    recall = []
    f1 = []
    for train,test in kfold.split(X,y):
        model = pipeline.fit(X[train],y[train])
        prediction = model.predict(X[test])
        scores = model.score(X[test],y[test])
        
        acc.append(scores*100)
        precision.append(precision_score(y[test],prediction,average = average_method)*100)
        print('                 POSITIVE | NEGATIVE')
        print('precision:    ',precision_score(y[test],prediction,average = None))
        recall.append(recall_score(y[test],prediction,average = average_method)*100)
        print('recall:       ',recall_score(y[test],prediction,average = None))
        f1.append(f1_score(y[test],prediction,average = average_method)*100)
        print('f1:           ',f1_score(y[test],prediction,average = None))
        print('-'*50)
        
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(acc), np.std(acc)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))



##-------------------------------------------------------------
##-------------------------------------------------------------
##                         MODEL-1
##                         MODEL with stopwords...
##                         ngram_range---(1,3)
##                         LogisticRegression
##-------------------------------------------------------------
##-------------------------------------------------------------

cvec = CountVectorizer(stop_words = None,max_features = 100000,ngram_range = (1,3))
lr = LogisticRegression()
line1 = Pipeline([
    ('count-vectorizer',cvec),
    ('Logistic Regression',lr)
])
cross_validation(splits = 5,X = train.clean_text,y = train.label,pipeline = line1,average_method = 'macro')

##-------------------------------------------------------------
##-------------------------------------------------------------
##                         MODEL-2
##                         MODEL with stopwords...
##                         ngram_range---(1,3)
##                         Balanced Random Forest
##-------------------------------------------------------------
##-------------------------------------------------------------
rf = RandomForestClassifier(n_estimators = 500,max_depth = 3,class_weight = 'balanced')
##pipeline
line2 = Pipeline([
    ('count-vectorizer',cvec),
    ('balanced_random_forest_classifier',rf)
])
cross_validation(splits = 5,X = train.clean_text,y = train.label,pipeline = line2,average_method = 'macro')
##-------------------------------------------------------------
##-------------------------------------------------------------
##                         MODEL-3
##                         MODEL with stopwords...
##                         ngram_range---(1,3)
##                         Model with TFIDF vectorizer and logistic Regression
##-------------------------------------------------------------
##-------------------------------------------------------------
tvec = TfidfVectorizer(stop_words = None,max_features = 100000,ngram_range = (1,3))

line3 = Pipeline([
    ('Tfidf-Vectorizer',tvec),
    ('logistic_regression',lr)
])

cross_validation(5,train.clean_text,train.label,line3,'macro')
##-------------------------------------------------------------
##-------------------------------------------------------------
##                         MODEL-4
##                         MODEL with stopwords...
##                         ngram_range---(1,3)
##                         Model with TFIDF vectorizer and randomforest
##-------------------------------------------------------------
##-------------------------------------------------------------
line4 = Pipeline([
    ('Tfidf-Vectorizer',tvec),
    ('rf',rf)
])

cross_validation(5,train.clean_text,train.label,line4,'macro')

##-------------------------------------------------------------
##-------------------------------------------------------------
##                         MODEL-5
##                         MODEL with stopwords...
##                         ngram_range---(1,3)
##                         Model with CountVectorizer and Logistic Regression and SMOTE                       
##-------------------------------------------------------------
##-------------------------------------------------------------
cv = CountVectorizer(stop_words = None,max_features = 100000,ngram_range = (1,3))
cv_text = cv.fit_transform(df.clean_text)

##partitioning 
y = df.label.values
X_train,X_test,y_train,y_test = train_test_split(cv_text,y,stratify = y,random_state = 42)


smt = SMOTE(random_state = 42,k_neighbors=3)
smt_train,smt_test = smt.fit_sample(X_train,y_train)

##fitting logistic_regression..
line5 = Pipeline([('smt_logistic_regression',lr)])

cross_validation(5,smt_train,smt_test,line5,'macro')

##prediction..
smt_pred = line5.predict(X_test)

confusion_matrix(y_test,smt_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,smt_pred)

##-------------------------------------------------------------
##-------------------------------------------------------------
##                         MODEL-6
##                         MODEL with stopwords...
##                         ngram_range---(1,3)
##                         Model with TfIdfVectorizer and Logistic Regression and SMOTE                       
##-------------------------------------------------------------
##-------------------------------------------------------------
##fitting count-vectorizer to the dataset..
tf = TfidfVectorizer(stop_words = None,max_features = 100000,ngram_range = (1,3))
tf_text = tf.fit_transform(df.clean_text)

##partitioning 
#from sklearn.model_selection import train_test_split
y = df.label.values
#X_train,X_test,y_train,y_test = train_test_split(tf_text,y,stratify = y,random_state = 42)


smt = SMOTE(random_state = 42,k_neighbors=3)
smt_train,smt_y = smt.fit_sample(tf_text,y)

##line6
final = Pipeline([('smt_logistic_regression',rf)])

cross_validation(5,smt_train,smt_y,final,'macro')

tf_text_test = tf.transform(test.clean_text)

pred = final.predict(tf_text_test)

sub_file = pd.DataFrame({'id':test['id'],
                        'label':pred})
sub_file.head()

sub = sub_file.to_csv('sub1.csv',index = None,header = True)

##-------------------------------------------------------------
##-------------------------------------------------------------
##                         MODEL-6
##                         MODEL with stopwords...
##                         ngram_range---(1,3)
##                         Model with TfIdfVectorizer and Logistic Regression and ADYSN                       
##-------------------------------------------------------------
##-------------------------------------------------------------

from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state = 42,n_neighbors = 4)

ada_train,ada_test = ada.fit_sample(X_train,y_train)


##line-7
line7 = Pipeline([('ada_logistic_regression',lr)])

cross_validation(5,ada_train,ada_test,line7,'macro')

ada_pred = line7.predict(X_test)
accuracy_score(y_test,ada_pred)







