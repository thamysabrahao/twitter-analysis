import spacy
import streamlit as st
from predictions_functions import *
from Services.wordcloud import *
from stream_portuguese_tweet import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
st.set_option('deprecation.showPyplotGlobalUse', False)

train_positive_text = open("Services/csv/positive_text.txt", "r").read()
train_negative_text = open("Services/csv/negative_text.txt", "r").read()


def pie_plot(df):
   df_pie = df['sentiment'].value_counts(normalize=True).reset_index()
   df_pie.columns = ['sentiment', 'Tweets percentage']
   df_pie['sentiment'] = df_pie['sentiment'].replace(0, 'negative').replace(1, 'positive')

   df_graph = pd.DataFrame()
   df_graph['sentiment'] = ['positive', 'negative']
   df_pie = pd.merge(df_graph, df_pie, on='sentiment', how='left').replace(np.nan, 0)

   df_pie['Tweets percentage'] = round(df_pie['Tweets percentage'], 2)
   labels = df_pie['sentiment'].values
   sizes = df_pie['Tweets percentage'].values
   explode = (0, 0.1)
   fig1, ax1 = plt.subplots()
   ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
   ax1.axis('equal')

   return plt.show()


# Add title on the page
st.title("Sentiment Analysis - Tweets written in portuguese")

# Ask user for input text
input_sent = st.text_input("Query to search @Twitter", "Your query goes here")

df = pd.DataFrame([])

if input_sent != 'Your query goes here':
    try:
       result = get_twitter_data(input_sent)
       df['text'] = result['text']
       cleaned_result = cleaned_data(result)
       pred = predictions(cleaned_result)
       df['sentiment'] = np.transpose(pred)

       st.pyplot(pie_plot(df))

       st.markdown("# Dataset analysis")
       st.markdown("**Positive sentiments**")
       st.pyplot(generate_wordcloud_df(df, 1))

       st.markdown("**Negative sentiments**")
       st.pyplot(generate_wordcloud_df(df, 0))

    except:
        pass