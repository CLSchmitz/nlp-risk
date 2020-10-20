from __future__ import unicode_literals, print_function
import thinc.extra.datasets
import random
import pandas as pd 
import plac
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import en_core_web_md

def load_data(split=0.8):

    df = pd.read_csv('../data/processed_labeled/df_final.csv', sep = '|')

    #shuffle
    df = df.sample(frac=1, random_state = 1000).reset_index(drop=True)

    texts = list(df.article_text_clean.values)
    cats = [{'RISK': bool(x), 'NO RISK': not bool(x)} for x in df.risk.values]

    split = int(len(texts) * split)

    return (texts[:split], cats[:split]), (texts[split:], cats[split:])

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

# Load the model
print('Loading Pretrained Spacy Model...')
nlp = en_core_web_md.load()

# Add the text classifier to the pipeline
textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})
nlp.add_pipe(textcat, last=True)

# Add label to text classifier
textcat.add_label("RISK")
textcat.add_label("NO RISK")
print('Successful')

# Load and format the labeled training data
print('Loading Data...')
(train_texts, train_cats), (dev_texts, dev_cats) = load_data()
train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
print('Successful')

# Get names of other pipes to disable them during training
pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*other_pipes):  # Disable those pipes
    
    optimizer = nlp.begin_training()

    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F")) #print loss after each epoch
    batch_sizes = compounding(4.0, 32.0, 1.001)
    
    # Up to ten epochs of training
    for i in range(10):
        losses = {}
        # Batch up the examples using spaCy's minibatch
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        with textcat.model.use_params(optimizer.averages):
            # Evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
        print(
            "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # Print a simple table
                losses["textcat"],
                scores["textcat_p"],
                scores["textcat_r"],
                scores["textcat_f"],
            )
        )
        if scores['textcat_f'] >= 0.8: 
            print('Reached validation F1-Score of over 0.8 in ' + str(i+1) + ' epochs. Stopping.')
            break

print('Saving Model...')
with nlp.use_params(optimizer.averages):
    nlp.to_disk('../models/spacy_text_classifier')
print('Saved Model')






