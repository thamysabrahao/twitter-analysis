from Services.model_functions import *
from Services.data_cleansing import *

# Loading trained models
count_vectorizer = load_model('ml-model/count_vectorizer.pickle')
model = load_model('ml-model/random_forest.pickle')

def cleaned_data(df):

    if df.shape[1] != 0:
        df['normalized_text'] = df['text'].apply(lambda column: string_normalize(column))
        df['clean_text'] = df['normalized_text'].apply(valid_text)
        X = list(df['clean_text'].astype(str))

        return X

def predictions(X):
    X_test = count_vectorizer.transform(X)
    pred = model.predict(X_test)

    return pred
