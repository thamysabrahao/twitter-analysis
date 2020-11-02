from Services.model_functions import *
from Services.data_cleansing import *

if __name__ == "__main__":

    df = read_csv('~/sentiment-portuguese-tweets/ml-model/Train100.csv.zip')
    df = df.sample(frac=1).reset_index(drop=True)

    print('normalize')
    df['normalized_text'] = df['tweet_text'].apply(lambda column: string_normalize(column))
    print('clean')
    df['clean_text'] = df['normalized_text'].apply(valid_text)

    positive_text = " ".join(df[df.sentiment == 1]['clean_text'].sum())
    with open("csv/positive_text.txt", "w") as file:
        file.write(positive_text)

    negative_text = " ".join(df[df.sentiment == 0]['clean_text'].sum())
    with open("csv/negative_text.txt", "w") as file:
        file.write(negative_text)
