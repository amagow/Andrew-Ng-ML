import numpy as np
import re
# from matplotlib import pyplot as plt
# from scipy import optimize
from scipy.io import loadmat
from sklearn import svm
# import math
from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize


def getVocabList():
    vocabList = np.genfromtxt("vocab.txt", dtype=object)
    return list(vocabList[:, 1].astype(str))

def processEmail(email_contents, verbose=True):
    vocabList = getVocabList()
    word_indices = []
    
    #Pre-email process
    
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    # hdrstart = email_contents.find(chr(10) + chr(10))
    # email_contents = email_contents[hdrstart:]
    
    # Lower case
    email_contents = email_contents.lower()
    
    #Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)
    
    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)
    
    
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    
    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
    
    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    
    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
    
    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]
        
    # Stem the email contents word by word
    stemmer = PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)

        if len(word) > 1:
            processed_email.append(word)
    
    for word in processed_email:
        for ii in range(len(vocabList)):
            if word == vocabList[ii]:
                word_indices.append(ii + 1)
    return word_indices

def emailFeatures(word_indices):
    n = 1899 #Number of words in the dictionary
    
    x = np.zeros(n)
    for idx in word_indices:
        x[idx - 1] = 1
    
    return x

#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

# Extract Features
with open('emailSample1.txt') as fid:
    file_contents = fid.read()

word_indices  = processEmail(file_contents)
features = emailFeatures(word_indices)

# print(len(features))
# print(sum(features))

#Train Linear SVM for Spam Classification

data = loadmat("spamTrain.mat")
Xtrain, ytrain = data['X'], data['y']
ytrain = ytrain.reshape(-1)

model = svm.LinearSVC(C=0.000045).fit(Xtrain, ytrain)

trainPred = model.predict(Xtrain)

print('Train Accuracy: {}'.format(np.mean(trainPred == ytrain) * 100))

dataTest = loadmat("spamTest.mat")
Xtest, ytest = dataTest['Xtest'], dataTest['ytest']

testPred = model.predict(Xtest)
print('Test Accuracy: {}'.format(np.mean(testPred == ytest) * 100))

# Get top 15 words
vocabList = getVocabList()

weights = model.coef_
weights = weights.reshape(-1)
weights = weights.argsort()[::-1]

words = []
for i in range(15):
    words.append(vocabList[weights[i] + 1])

print(words)

with open('emailSample2.txt') as fid:
    file_contents = fid.read()

word_indices  = processEmail(file_contents)
features = emailFeatures(word_indices)
print(model.predict(features.reshape(-1,1).T))
