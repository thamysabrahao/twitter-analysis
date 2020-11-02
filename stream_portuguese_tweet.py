import tweepy
import pandas as pd
import json

print('started!')

# Don't forget to add your own Twitter's API access to config dict
config = {
    "consumer_key":       "yours_key_here",
    "consumer_secret":    "yours_secret_here",
    "oauth_token":        "yours_token_here",
    "oauth_token_secret": "yours_token_secret_here"
}

consumer_key = config["consumer_key"]
consumer_secret = config["consumer_secret"]
oauth_token = config["oauth_token"]
oauth_token_secret = config["oauth_token_secret"]


def get_twitter_data(query):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(oauth_token, oauth_token_secret)
    twitter = tweepy.API(auth)
    results = twitter.search(q=str(query), lang='pt')

    df = pd.DataFrame([])
    text = []
    for tweet in results:
        text.append(tweet.text)

    df['text'] = text

    return df





