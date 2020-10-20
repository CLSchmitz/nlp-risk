import pandas as pd
import glob
import re
#import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
#nltk.download('stopwords')
#nltk.download('wordnet')

# This script takes in the 2000+ CSV's from the RPA, cleans them, and joins them into one dataframe.


### Read and Concatenate Files; Drop NA's
path = '../data'
all_files = glob.glob(path + "/raw/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, sep = '|', header=0)
    li.append(df)
df = pd.concat(li, axis=0, ignore_index=True).dropna()


### Functions and tools to clean text
stop = stopwords.words('english')
ps = PorterStemmer()

def clean(t):
'''
Returns a cleaned version of the string, t, that is entered.
'''
    
    t = remove_between_brackets(t) #remove text between brackets
    t = ''.join([c for c in t if c in string.ascii_letters + ' .']) #remove everything but alphabet, spaces, and full stops
    t = t.lower() #lowercase everything

    return t

def stem(t):

    t = remove_between_brackets(t) #remove text between brackets
    t = ''.join([c for c in t if c in string.ascii_letters + ' ']) #remove everything but alphabet and spaces
    t = t.lower().split() #lowercase everything and split into tokens
    t = [c for c in t if c not in stop] #remove stopwords
    t = ' '.join([ps.stem(c) for c in t if c not in stop]) #stem words

    return t
   
def remove_between_brackets(dirty):
'''
Removes all content in 'dirty', a string, that is between curly brackets (common in web content from RPA).
'''

  clean = ""
  bad = 0

  for c in dirty:
     if c == '{':
        bad += 1
     elif c == '}':
        bad -= 1

     if bad == 0:
        clean += c

  return clean

### Apply above cleaning functions to text, add cleaned texts to dataframe, and save new dataset
df['article_text_clean'] = df['article_text'].apply(lambda x: clean(x))
df['article_text_stemmed'] = df['article_text'].apply(lambda x: stem(x))
df.to_csv(path + '/processed/data_clean.csv', sep = '|')