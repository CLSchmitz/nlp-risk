import spacy
import pandas as pd
import glob 

# Load trained model
nlp = spacy.load('../models/spacy_text_classifier')

# Load real supplier samples
path = '../data/test'
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, sep = '|', header = 0)
    print(df)
    li.append(df)
df = pd.concat(li, axis=0, ignore_index=True)

    # # test the saved model
    # print("Loading from", output_dir)
    # nlp2 = spacy.load(output_dir)
    # doc2 = nlp2(test_text)
    # print(test_text, doc2.cats)