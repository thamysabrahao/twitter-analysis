import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
nltk.download('stopwords')

def generate_wordcloud(text):

    sw = stopwords.words('portuguese')
    new_stopwords = ['sim', 'nao', 'https', 'www', 'rt']
    sw.extend(new_stopwords)

    wordcloud = WordCloud(collocations=False,
                          background_color='white',
                          stopwords=sw).generate(text)

    plt.imshow(wordcloud)
    plt.axis("off")

    return plt.show()


def generate_wordcloud_df(df, sentiment:int):

    text = " ".join(df[df.sentiment == sentiment]['text'])
    sw = stopwords.words('portuguese')
    new_stopwords = ['sim', 'nao', 'https', 'www', 'rt']
    sw.extend(new_stopwords)

    wordcloud = WordCloud(collocations=False,
                          background_color='white',
                          stopwords=sw).generate(text)

    plt.imshow(wordcloud)
    plt.axis("off")

    return plt.show()