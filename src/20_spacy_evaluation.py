import spacy
import pandas as pd
import glob 
import string

# Load trained model
nlp = spacy.load('../models/spacy_text_classifier')

# Load real supplier samples
df = pd.read_excel('../data/test/domain_samples.xlsx')

# Same cleaning process as training data
def clean(t):

    t = remove_between_brackets(t)
    t = ''.join([c for c in t if c in string.ascii_letters + ' .'])
    t = t.lower()

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

# Clean data and add to new column
df['article_text_clean'] = df.article_text.apply(lambda x: clean(x))

# Use model to make predictions and save to new column
df['predictions'] = df.article_text_clean.apply(lambda x: nlp(x).cats)

# Save predictions to .xlsx
df.to_excel('../results/test_evals.xlsx')