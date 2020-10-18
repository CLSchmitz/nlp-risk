import pandas as pd
import glob
import re
#import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
#nltk.download('stopwords')
#nltk.download('wordnet')


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

    t = remove_between_brackets(t)
    t = ''.join([c for c in t if c  in string.ascii_letters + ' '])
    t = t.lower().split()
    t = ' '.join([ps.stem(c) for c in t if c not in stop])

    return t

def remove_between_brackets(dirty):

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

df['article_text_clean'] = df['article_text'].apply(lambda x: clean(x))

df.to_csv(path + '/processed/data_clean.csv', sep = '|')