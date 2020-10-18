import pandas as import pd
import gensim

df = pd.read_csv('../data/processed/data_clean.csv', sep = '|')
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)