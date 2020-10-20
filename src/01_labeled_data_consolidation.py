import pandas as pd 
import glob

# This script combines the four separate labeled data files into one final file, ready for analysis.

path = '../data'
all_files = glob.glob(path + "/processed_labeled/labeled_by_person/*.xlsx")

li = []
for filename in all_files:
    df = pd.read_excel(filename, header=0)
    print(filename)
    print(df.columns)
    print(df.shape)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True).dropna()
df_unlabelled = pd.read_csv(path + '/processed/data_clean.csv', sep = '|')

df = df.drop(columns = ['article_text_clean'])

df_final = df.merge(df_unlabelled[['Unnamed: 0', 'article_text_clean', 'article_text_stemmed']], how = 'left', left_on = 'Column1', right_on = 'Unnamed: 0')
df_final = df_final.drop(columns = ['Column1', 'Unnamed: 0'])

df_final.to_csv(path + '/processed_labeled/df_final.csv', sep = '|')