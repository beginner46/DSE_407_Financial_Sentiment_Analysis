#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import re
import numpy as np
import pandas as pd

# text treatement
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

import warnings
warnings.filterwarnings("ignore")

import gensim
from gensim.models import Word2Vec

import multiprocessing

from nltk.stem import WordNetLemmatizer

import csv,sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest,chi2 


# In[2]:


dataset_train = pd.read_csv('project2_training_data.tsv', delimiter = '\t', quoting = 3)
dataset_labels = pd.read_csv('project2_training_data_labels.tsv', delimiter = '\t', quoting = 3)


# In[3]:


dataset_train = dataset_train.dropna(axis='columns', inplace = False)
dataset_labels = dataset_labels.dropna(axis='columns', inplace = False)


# In[4]:


dataset_train.head()


# In[5]:


dataset_labels.head()


# In[6]:


dataset_train.shape


# In[7]:


dataset_labels.shape


# In[8]:


result = pd.concat([dataset_train, dataset_labels], axis=1, join='inner')


# In[9]:


result.head()


# In[10]:


result = result.rename_axis(None, axis=1)


# In[11]:


result.head()


# In[12]:


result.columns = ['Texts', 'Sentiment']


# In[13]:


result.head()


# In[14]:


df = result.copy()


# In[15]:


df = df.dropna(axis='index', inplace=False)


# In[16]:


df.head()


# In[17]:


data_expo = df.copy()


# ## Functions

# In[18]:


def sw_removal(data):
    stop = set(stopwords.words('englsih'))
    data = data.apply(lambda x: [word for word in x if word not in stop])
    
# def port_stem(data):
#     data = data.apply(lambda x: [PorterStemmer.stem(word, to_lowercase = False) for word in x])
    
    
def lemmatize_word_helper(text):
    lemmatizer  = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word) for word in text]
    return lemma

def lemmatize(data):
    data = data.apply(lambda x: lemmatize_word_helper(x))


# ## Exploring the distribution
# 

# In[19]:


cnt_pro = data_expo['Sentiment'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Srntiment', fontsize=12)
plt.xticks(rotation=90)
plt.show();


# In[20]:


number_of_words = 0

from bs4 import BeautifulSoup ##code to remove punctuations and symbols
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
data_expo['Texts'] = data_expo['Texts'].apply(cleanText)

for i in range(0,1810):
    lines = data_expo['Texts'][i].split()
    number_of_words += len(lines)
    
print(f"count of words {number_of_words}")

results = set()
unique = data_expo['Texts'].str.lower().str.split().apply(results.update)
number_of_unique_words = len(results)
print(f"count of unique words {number_of_unique_words}")


# ## Cleaning the data for removing punctuations and symbols

# In[21]:


from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['Texts'] = df['Texts'].apply(cleanText)


# In[22]:


lmtzr = WordNetLemmatizer()
df['Lemmas_lst'] = df['Texts'].apply(lambda lst:[lmtzr.lemmatize(word) for word in lst])
stemmer = PorterStemmer()
df['Stemmed_lst'] = df['Texts'].apply(lambda x: [stemmer.stem(y) for y in x])
df['Lemmas'] = [' '.join(map(str, l)) for l in df['Lemmas_lst']]
df['Stemmed'] = [' '.join(map(str, l)) for l in df['Stemmed_lst']]


# In[23]:


df.head()


# ## The classifier Part

# In[ ]:



opt1=input('Enter\n\t "a" for Classification after tfidf vectorisation \n\t "b" for Classification with Countvectoriser vectorisation \n\t "c" for Classification after tfidf vectorisation and stemming \n\t "d" for Classification after Countvectoriser vectorisation and stemming \n\t "e" for Classification after tfidf vectorisation and Lemmatization \n\t "f" for Classification after Countvectoriser vectorisation and Lemmatization \n\t "q" to quit \n')



if opt1=='a':            # simple run with no parameter tuning
#     clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
# #    clf = RandomForestClassifier(criterion='gini',class_weight='balanced') 
    
#     vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,3),token_pattern=r'\b\w+\b')
#     tfidf = vectorizer.fit_transform(data)
#     terms=vectorizer.get_feature_names()
#     tfidf = tfidf.toarray()

#     # Training and Test Split
    
#     trn_data, tst_data, trn_cat, tst_cat = train_test_split(tfidf, labels, test_size=0.20, random_state=42,stratify=labels)   
    
#     #Classificaion    
#     clf.fit(trn_data,trn_cat)
#     predicted = clf.predict(tst_data)
#     predicted =list(predicted)


    
    data = df['Texts']
    labels = df['Sentiment']

    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }


    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigra
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)


##################################################### Countvectoriser ###################################################    
    
    

elif opt1=='b':         
    # Training and Test Split
    data = df['Texts']
    labels = df['Sentiment']
    
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }
        

    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
    
    ##################################################### Tfidf vectoriser with lemmatisation ###################################################    
    

elif opt1=='e':     
    
   
    # Training and Test Split
    
    data = df['Lemmas']
    
    labels = df['Sentiment']
    
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }
        

    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
   
    
        ##################################################### Tfidf vectoriser with stemming ###################################################    
    

elif opt1=='c':     
    
   
    # Training and Test Split
    
    data = df['Stemmed']
    
    labels = df['Sentiment']
    
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }
        

    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
   
    
    
    
   ##################################################### Countvectoriser with lemmatisation ###################################################    
    

elif opt1=='d':     
    
   
    # Training and Test Split
    #df['Texts'] = lemmatize(df['Texts'])
    
    data = df['Lemmas']
   
    labels = df['Sentiment']
    
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }
        

    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
     
       ##################################################### Countvectoriser with stemming ###################################################    
    

elif opt1=='f':     
    
   
    # Training and Test Split
    #df['Texts'] = lemmatize(df['Texts'])
    
    data = df['Stemmed']
   
    labels = df['Sentiment']
    
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
    opt2 = input("Choose a classifier : "
                   "\n\n\t 'lr' to select logistic regression" 
                   "\n\t 'ls' to select Linear SVC" 
                   "\n\t 's' to select SVM" 
                   "\n\t 'dt' to select Decision Tree"   
                   "\n\t 'rf' to select Random Forest"
                   "\n\t 'mn' to select multinomial naive bayes \n\n")   
# Naive Bayes Classifier    
    if opt2=='mn':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif opt2=='ls': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif opt2=='s':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif opt2=='lr':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif opt2=='dt':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
# Random Forest Classifier    
    elif opt2=='rf':
        clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
        clf_parameters = {
                    'clf__criterion':('gini', 'entropy'), 
                    'clf__max_features':('auto', 'sqrt', 'log2'),   
                    'clf__n_estimators':(30,50,100,200),
                    'clf__max_depth':(10,20,),
                    }
        

    else:
        print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
        sys.exit(0)                                  
# Feature Extraction
    pipeline = Pipeline([
    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k='all')),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_macro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
     




elif opt1 == 'q':
    print('Bye!\n')
    sys.exit(0)
    
else:
    print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
    sys.exit(0)

################################################ Evaluation ############################################################
print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  

pr=precision_score(tst_cat, predicted, average='macro') 
print ('\n Precision:'+str(pr)) 

rl=recall_score(tst_cat, predicted, average='macro') 
print ('\n Recall:'+str(rl))

fm=f1_score(tst_cat, predicted, average='macro') 
print ('\n Macro Averaged F1-Score:'+str(fm))

# Evaluation
print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  

pr=precision_score(tst_cat, predicted, average='micro') 
print ('\n Precision:'+str(pr)) 

rl=recall_score(tst_cat, predicted, average='micro') 
print ('\n Recall:'+str(rl))

fm=f1_score(tst_cat, predicted, average='micro') 
print ('\n Micro Averaged F1-Score:'+str(fm))


# In[ ]:




