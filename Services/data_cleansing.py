from nltk import word_tokenize
from nltk.corpus import stopwords
import spacy
import unicodedata
import string
import numpy as np
import nltk
nltk.download('stopwords')

nlp = spacy.load("pt_core_news_sm")

def string_normalize(text):
    # function to 'normalize' words and remove punctuation
    if type(text) is str:
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore') \
            .decode('utf-8').lower().replace('\n', ' ').replace('\r', ' ').replace(r'[^\w\s]', '').replace('https', '')

        table = str.maketrans({key: None for key in string.punctuation})

        return text.translate(table)

    else:
        return np.nan


def valid_text(text):
    text = nlp(str(text))

    # lemmatization
    lemma_msgn = [y.lemma_ for y in text]

    # removing stopwords
    sw = stopwords.words('portuguese')
    msg_wo_sw = [word for word in lemma_msgn if word not in sw]

    return msg_wo_sw