''':type
regionCode : https://www.iso.org/iso-3166-country-codes.html
relevanceLanguage : https://www.loc.gov/standards/iso639-2/php/code_list.php
google_api_searchlist : https://developers.google.com/youtube/v3/docs/search/list

anomaly detection
https://towardsdatascience.com/time-series-of-price-anomaly-detection-13586cd5ff46
https://neptune.ai/blog/anomaly-detection-in-time-series#:~:text=What%20are%20anomalies%2Foutliers%20and,generated%20by%20a%20different%20mechanism.%E2%80%9D

Emoji_dictionary : https://unicode.org/emoji/charts-13.0/full-emoji-list.html
'''
# pip install google-api-python-client

# youtube class
from googleapiclient.discovery import build #API
from datetime import datetime
import pandas as pd
import numpy as np

# analysis class
from matplotlib import pyplot as plt
import matplotlib as mpl # Seasonal plot
import matplotlib.gridspec as gridspec #Splitplot
import seaborn as sns #Graph Visualization
import colored
from sklearn.decomposition import PCA #Anomaly
from sklearn.preprocessing import StandardScaler #Anomaly
from sklearn.ensemble import IsolationForest #Anomaly

# sentiment analysis
import re
import string
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet #lemmatization
from nltk.stem import WordNetLemmatizer #lemmatization
from collections import Counter #frequent/rare works
from textblob import TextBlob #polarity
# pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification #BERT
import torch #BERT

def remove_cleantext(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower() # Lower
    text = re.sub('\[.*?\]', '', text) # SquareBracket
    text = re.sub('https?://\S+|www\.\S+', '', text) # URL
    text = re.sub('<.*?>+', '', text) # Bracket
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Punctuation
    text = re.sub('\n', ' ', text) #
    text = re.sub('(?<=#)\w+', '', text) # Hash
    text = re.sub('[\w\.-]+@[\w\.-]+', '', text) # Email
    text = re.sub('[0-9]+', '', text) # Number
    text = text.strip()
    #text = text.strip() # Remove Space
    #text = text.split() # Remove Space
    return text

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001F3C6" # trophy
                           u"\U0001F947" # trophy                               
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_emoticons(text):
    rem_emoticons = {
        u":‑\)": "Happy face or smiley",
        u":\)": "Happy face or smiley",
        u":-\]": "Happy face or smiley",
        u":\]": "Happy face or smiley",
        u":-3": "Happy face smiley",
        u":3": "Happy face smiley",
        u":->": "Happy face smiley",
        u":>": "Happy face smiley",
        u"8-\)": "Happy face smiley",
        u":o\)": "Happy face smiley",
        u":-\}": "Happy face smiley",
        u":\}": "Happy face smiley",
        u":-\)": "Happy face smiley",
        u":c\)": "Happy face smiley",
        u":\^\)": "Happy face smiley",
        u"=\]": "Happy face smiley",
        u"=\)": "Happy face smiley",
        u":‑D": "Laughing, big grin or laugh with glasses",
        u":D": "Laughing, big grin or laugh with glasses",
        u"8‑D": "Laughing, big grin or laugh with glasses",
        u"8D": "Laughing, big grin or laugh with glasses",
        u"X‑D": "Laughing, big grin or laugh with glasses",
        u"XD": "Laughing, big grin or laugh with glasses",
        u"=D": "Laughing, big grin or laugh with glasses",
        u"=3": "Laughing, big grin or laugh with glasses",
        u"B\^D": "Laughing, big grin or laugh with glasses",
        u":-\)\)": "Very happy",
        u":‑\(": "Frown, sad, andry or pouting",
        u":-\(": "Frown, sad, andry or pouting",
        u":\(": "Frown, sad, andry or pouting",
        u":‑c": "Frown, sad, andry or pouting",
        u":c": "Frown, sad, andry or pouting",
        u":‑<": "Frown, sad, andry or pouting",
        u":<": "Frown, sad, andry or pouting",
        u":‑\[": "Frown, sad, andry or pouting",
        u":\[": "Frown, sad, andry or pouting",
        u":-\|\|": "Frown, sad, andry or pouting",
        u">:\[": "Frown, sad, andry or pouting",
        u":\{": "Frown, sad, andry or pouting",
        u":@": "Frown, sad, andry or pouting",
        u">:\(": "Frown, sad, andry or pouting",
        u":'‑\(": "Crying",
        u":'\(": "Crying",
        u":'‑\)": "Tears of happiness",
        u":'\)": "Tears of happiness",
        u"D‑':": "Horror",
        u"D:<": "Disgust",
        u"D:": "Sadness",
        u"D8": "Great dismay",
        u"D;": "Great dismay",
        u"D=": "Great dismay",
        u"DX": "Great dismay",
        u":‑O": "Surprise",
        u":O": "Surprise",
        u":‑o": "Surprise",
        u":o": "Surprise",
        u":-0": "Shock",
        u"8‑0": "Yawn",
        u">:O": "Yawn",
        u":-\*": "Kiss",
        u":\*": "Kiss",
        u":X": "Kiss",
        u";‑\)": "Wink or smirk",
        u";\)": "Wink or smirk",
        u"\*-\)": "Wink or smirk",
        u"\*\)": "Wink or smirk",
        u";‑\]": "Wink or smirk",
        u";\]": "Wink or smirk",
        u";\^\)": "Wink or smirk",
        u":‑,": "Wink or smirk",
        u";D": "Wink or smirk",
        u":‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"X‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"=p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑/": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u":[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=/": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=L": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
        u":‑\|": "Straight face",
        u":\|": "Straight face",
        u":$": "Embarrassed or blushing",
        u":‑x": "Sealed lips or wearing braces or tongue-tied",
        u":x": "Sealed lips or wearing braces or tongue-tied",
        u":‑#": "Sealed lips or wearing braces or tongue-tied",
        u":#": "Sealed lips or wearing braces or tongue-tied",
        u":‑&": "Sealed lips or wearing braces or tongue-tied",
        u":&": "Sealed lips or wearing braces or tongue-tied",
        u"O:‑\)": "Angel, saint or innocent",
        u"O:\)": "Angel, saint or innocent",
        u"0:‑3": "Angel, saint or innocent",
        u"0:3": "Angel, saint or innocent",
        u"0:‑\)": "Angel, saint or innocent",
        u"0:\)": "Angel, saint or innocent",
        u":‑b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"0;\^\)": "Angel, saint or innocent",
        u">:‑\)": "Evil or devilish",
        u">:\)": "Evil or devilish",
        u"\}:‑\)": "Evil or devilish",
        u"\}:\)": "Evil or devilish",
        u"3:‑\)": "Evil or devilish",
        u"3:\)": "Evil or devilish",
        u">;\)": "Evil or devilish",
        u"\|;‑\)": "Cool",
        u"\|‑O": "Bored",
        u":‑J": "Tongue-in-cheek",
        u"#‑\)": "Party all night",
        u"%‑\)": "Drunk or confused",
        u"%\)": "Drunk or confused",
        u":-###..": "Being sick",
        u":###..": "Being sick",
        u"<:‑\|": "Dump",
        u"\(>_<\)": "Troubled",
        u"\(>_<\)>": "Troubled",
        u"\(';'\)": "Baby",
        u"\(\^\^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(\^_\^;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(-_-;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(~_~;\) \(・\.・;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(-_-\)zzz": "Sleeping",
        u"\(\^_-\)": "Wink",
        u"\(\(\+_\+\)\)": "Confused",
        u"\(\+o\+\)": "Confused",
        u"\(o\|o\)": "Ultraman",
        u"\^_\^": "Joyful",
        u"\(\^_\^\)/": "Joyful",
        u"\(\^O\^\)／": "Joyful",
        u"\(\^o\^\)／": "Joyful",
        u"\(__\)": "Kowtow as a sign of respect, or dogeza for apology",
        u"_\(\._\.\)_": "Kowtow as a sign of respect, or dogeza for apology",
        u"<\(_ _\)>": "Kowtow as a sign of respect, or dogeza for apology",
        u"<m\(__\)m>": "Kowtow as a sign of respect, or dogeza for apology",
        u"m\(__\)m": "Kowtow as a sign of respect, or dogeza for apology",
        u"m\(_ _\)m": "Kowtow as a sign of respect, or dogeza for apology",
        u"\('_'\)": "Sad or Crying",
        u"\(/_;\)": "Sad or Crying",
        u"\(T_T\) \(;_;\)": "Sad or Crying",
        u"\(;_;": "Sad of Crying",
        u"\(;_:\)": "Sad or Crying",
        u"\(;O;\)": "Sad or Crying",
        u"\(:_;\)": "Sad or Crying",
        u"\(ToT\)": "Sad or Crying",
        u";_;": "Sad or Crying",
        u";-;": "Sad or Crying",
        u";n;": "Sad or Crying",
        u";;": "Sad or Crying",
        u"Q\.Q": "Sad or Crying",
        u"T\.T": "Sad or Crying",
        u"QQ": "Sad or Crying",
        u"Q_Q": "Sad or Crying",
        u"\(-\.-\)": "Shame",
        u"\(-_-\)": "Shame",
        u"\(一一\)": "Shame",
        u"\(；一_一\)": "Shame",
        u"\(=_=\)": "Tired",
        u"\(=\^\·\^=\)": "cat",
        u"\(=\^\·\·\^=\)": "cat",
        u"=_\^=	": "cat",
        u"\(\.\.\)": "Looking down",
        u"\(\._\.\)": "Looking down",
        u"\^m\^": "Giggling with hand covering mouth",
        u"\(\・\・?": "Confusion",
        u"\(?_?\)": "Confusion",
        u">\^_\^<": "Normal Laugh",
        u"<\^!\^>": "Normal Laugh",
        u"\^/\^": "Normal Laugh",
        u"\（\*\^_\^\*）": "Normal Laugh",
        u"\(\^<\^\) \(\^\.\^\)": "Normal Laugh",
        u"\(^\^\)": "Normal Laugh",
        u"\(\^\.\^\)": "Normal Laugh",
        u"\(\^_\^\.\)": "Normal Laugh",
        u"\(\^_\^\)": "Normal Laugh",
        u"\(\^\^\)": "Normal Laugh",
        u"\(\^J\^\)": "Normal Laugh",
        u"\(\*\^\.\^\*\)": "Normal Laugh",
        u"\(\^—\^\）": "Normal Laugh",
        u"\(#\^\.\^#\)": "Normal Laugh",
        u"\（\^—\^\）": "Waving",
        u"\(;_;\)/~~~": "Waving",
        u"\(\^\.\^\)/~~~": "Waving",
        u"\(-_-\)/~~~ \($\·\·\)/~~~": "Waving",
        u"\(T_T\)/~~~": "Waving",
        u"\(ToT\)/~~~": "Waving",
        u"\(\*\^0\^\*\)": "Excited",
        u"\(\*_\*\)": "Amazed",
        u"\(\*_\*;": "Amazed",
        u"\(\+_\+\) \(@_@\)": "Amazed",
        u"\(\*\^\^\)v": "Laughing,Cheerful",
        u"\(\^_\^\)v": "Laughing,Cheerful",
        u"\(\(d[-_-]b\)\)": "Headphones,Listening to music",
        u'\(-"-\)': "Worried",
        u"\(ーー;\)": "Worried",
        u"\(\^0_0\^\)": "Eyeglasses",
        u"\(\＾ｖ\＾\)": "Happy",
        u"\(\＾ｕ\＾\)": "Happy",
        u"\(\^\)o\(\^\)": "Happy",
        u"\(\^O\^\)": "Happy",
        u"\(\^o\^\)": "Happy",
        u"\)\^o\^\(": "Happy",
        u":O o_O": "Surprised",
        u"o_0": "Surprised",
        u"o\.O": "Surpised",
        u"\(o\.o\)": "Surprised",
        u"oO": "Surprised",
        u"\(\*￣m￣\)": "Dissatisfied",
        u"\(‘A`\)": "Snubbed or Deflated"
    }
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in rem_emoticons) + u')')
    return emoticon_pattern.sub(r'', text)

# API Search list :
class youtube:
    def __init__(self, api):
        self.api = api

    def trend_video(self, year, month, day, after_day, n_max, region, language):
        # Region : MY, KR
        # Language : en, ko
        youtube = build('youtube', 'v3', developerKey=self.api)
        print(colored.fg('green'))
        print("MESSAGE : Your start year/month/day :", year, "/", month, "/", day)
        print("MESSAGE : Your end year/month/day :", year, "/", month, "/", day+after_day)
        print(colored.fg('white'))
        start_time = datetime(year=year, month=month, day=day).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = datetime(year=year, month=month, day=day+after_day).strftime('%Y-%m-%dT%H:%M:%SZ')

        res_trendvideo = youtube.search().list(part='snippet', type='video', order='viewCount', maxResults=n_max,
                                             regionCode=region, relevanceLanguage=language, publishedAfter=start_time,
                                             publishedBefore=end_time).execute()
        print(colored.fg('green'))
        print("MESSAGE : List of channel,video titles and channel id in top view counts")
        print(colored.fg('white'))

        trendvideo_list = []

        for index, item in enumerate(res_trendvideo['items']): # enumerate can show a sequence of loop
            dic = {'channelTitle':item['snippet']['channelTitle'], 'Video Title':item['snippet']['title'],
                   'ChannelID':item['snippet']['channelId']}
            trendvideo_list.append(dic)
            print(index, "|Channel Title :|", item['snippet']['channelTitle'], "\n", index, "|Video Title :|",
                  item['snippet']['title'], "\n", index, "|ChannelID :|", item['snippet']['channelId'])

        return trendvideo_list

    def popular_video(self, n_max, region):

        youtube = build('youtube', 'v3', developerKey=self.api)
        print(colored.fg('green'))
        print("MESSAGE : Your selected list of popular videos are :", n_max)
        print(colored.fg('white'))

        res_popular = youtube.videos().list(part='snippet', chart='mostPopular',
                                            maxResults=n_max, regionCode=region).execute()

        print(colored.fg('green'))
        print("MESSAGE : List of channel,video titles and channel id in top view counts")
        print(colored.fg('white'))

        popular_list = []

        for index, item in enumerate(res_popular['items']): # enumerate can show a sequence of loop
            dic = {'channelTitle':item['snippet']['channelTitle'], 'Video Title':item['snippet']['title'],
                   'ChannelID':item['snippet']['channelId'], 'publishedAt':item['snippet']['publishedAt']}
            popular_list.append(dic)
            print(index, "|Channel Title :|", item['snippet']['channelTitle'], "\n",
                  index, "|Video Title :|", item['snippet']['title'], "\n",
                  index, "|ChannelID :|", item['snippet']['channelId'], "\n",
                  index, "|publishedAt :|", item['snippet']['publishedAt'])

        return popular_list

    def get_channel_stats(self, channel_id, sort): #sort conditions : date, view, like, comment, dislike

        youtube = build('youtube', 'v3', developerKey=self.api)
        res_channel = youtube.channels().list(part='snippet', id=channel_id).execute()
        print(colored.fg('green'))
        print("MESSAGE : Your selected channel title is :", res_channel['items'][0]['snippet']['title'])
        print("MESSAGE : Channel published at :", res_channel['items'][0]['snippet']['publishedAt'])

        res_statistic = youtube.channels().list(part='statistics', id=channel_id).execute()
        print("MESSAGE : SubscriberCount :", res_statistic['items'][0]['statistics']['subscriberCount'])
        print("MESSAGE : VideoCount :", res_statistic['items'][0]['statistics']['videoCount'])

        res_content = youtube.channels().list(part='contentDetails', id=channel_id).execute()
        print("MESSAGE : Playlist ID :", res_content['items'][0]['contentDetails']['relatedPlaylists']['uploads'])
        playlist_id = res_content['items'][0]['contentDetails']['relatedPlaylists'][
            'uploads']  # Export Playlist ID. Playlist ID is different from Channel ID.

        videos = []
        next_page_token = None

        while 1:
            res = youtube.playlistItems().list(part='snippet', playlistId=playlist_id, maxResults=50,
                                               pageToken=next_page_token).execute()
            videos += res['items']
            next_page_token = res.get('nextPageToken')
            if next_page_token is None:
                break

        print("MESSAGE : Playlist Result Counts :", res['pageInfo']['totalResults'])
        print(colored.fg('white'))

        # Sort out videoIds only
        video_ids = list(map(lambda x: x['snippet']['resourceId']['videoId'], videos))
        #print(colored.fg('green'))
        #print("INPUT : Please write text to set your sort conditions")
        #print(colored.fg('white'))
        #sort = input("Conditions are : date, view, like, dislike :")
        print(colored.fg('green'))
        print("MESSAGE : You have selected sort condition as :", sort)
        print(colored.fg('white'))

        stats = []
        rating = []
        title = []
        summary = []

        for i in range(0, len(video_ids), 50):  # Sequence : 0~50, 50~100,,,
            res_stat = youtube.videos().list(id=','.join(video_ids[i:i + 50]), part='statistics').execute()
            res_rate = youtube.videos().list(id=','.join(video_ids[i:i + 50]), part='contentDetails').execute()
            res_title = youtube.videos().list(id=','.join(video_ids[i:i + 50]), part='snippet').execute()
            stats += res_stat['items']
            rating += res_rate['items']
            title += res_title['items']

        for i in range(0, len(stats)):
            title[i]['snippet']['publishedAt'] = datetime.strptime(title[i]['snippet']['publishedAt'],
                                                                   '%Y-%m-%dT%H:%M:%SZ')
            dic = {'title': title[i]['snippet']['title'], 'publishedAt': title[i]['snippet']['publishedAt'],
                   'videoId': stats[i]['id'],
                   'statistics': stats[i]['statistics'], 'contentRating': rating[i]['contentDetails']['contentRating'],
                   }
            summary.append(dic)

        if sort == 'dislike':
            summary = sorted(summary, key=lambda x: int(x['statistics']['dislikeCount']), reverse=True)
        elif sort == 'like':
            summary = sorted(summary, key=lambda x: int(x['statistics']['likeCount']), reverse=True)
        elif sort == 'view':
            summary = sorted(summary, key=lambda x: int(x['statistics']['viewCount']), reverse=True)
        elif sort == 'comment':
            summary = sorted(summary, key=lambda x: int(x['statistics']['commentCount']), reverse=True)
        else:
            summary = sorted(summary, key=lambda x: x['publishedAt'], reverse=True)

        return summary

# Analysis using selected channelID
class analysis:
    def __init__(self, scaler):
        self.scaler = int(scaler)

    def setup(self):
        #  general plots list
        global date
        date = []
        global view
        view = []
        global like
        like = []
        global dislike
        dislike = []
        global comment
        comment = []
        global title
        title = []
        global videoId
        videoId = []
        #  ratio plots list
        fbtotal = []  # like+dislike
        fbratio = []  # fbtotal/view
        likeratio = []  # like/fbtotal
        dislikeratio = []  # dislike/fbtotal

        for i in range(0, len(summary)):
            # All plots
            x = summary[i]['publishedAt']
            date.append(x)
            a = int(summary[i]['statistics']['viewCount']) / self.scaler
            view.append(a)
            b = int(summary[i]['statistics']['likeCount']) / self.scaler
            like.append(b)
            c = int(summary[i]['statistics']['dislikeCount']) / self.scaler
            dislike.append(c)
            d = int(summary[i]['statistics']['commentCount']) / self.scaler
            comment.append(d)
            extra_01 = str(summary[i]['title'])
            title.append(extra_01)
            extra_02 = str(summary[i]['videoId'])
            videoId.append(extra_02)

        for i in range(0, len(summary)):
            # Ratio plots
            e = like[i] + dislike[i]  # like+dislike
            fbtotal.append(e)

        for i in range(0, len(summary)):
            try:
                f = fbtotal[i] / view[i]  # fbtotal/view
            except ZeroDivisionError:
                f = 0
            fbratio.append(f)
            try:
                g = like[i] / fbtotal[i]  # like/fbtotal
            except ZeroDivisionError:
                g = 0
            likeratio.append(g)
            try:
                h = dislike[i] / fbtotal[i]  # dislike/fbtotal
            except ZeroDivisionError:
                h = 0
            dislikeratio.append(h)

        return fbtotal, fbratio, likeratio, dislikeratio

    def allchart(self):

        # Plot_001 : All plots
        sns.set()
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 14), dpi=80)
        ax1.plot(date, view, label='ViewCount')
        ax1.set_title("ViewCount", fontsize=15)
        ax1.set_ylabel('ViewCount')
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)

        ax2.plot(date, like, color='blue', label='LikeCount')
        ax2.set_title("LikeCount", fontsize=15)
        ax2.set_ylabel('LikeCount')
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)

        ax3.plot(date, dislike, color='red', label='DislikeCount')
        ax3.set_title("Dislike Count", fontsize=15)
        ax3.set_ylabel('Dislike Count')
        ax3.grid(True)
        ax3.tick_params(axis='x', rotation=45, labelsize=8)

        ax4.plot(date, comment, color='black', label='CommentCount')
        ax4.set_title("Comment Count", fontsize=15)
        ax4.set_ylabel('Comment Count')
        ax4.grid(True)
        ax4.tick_params(axis='x', rotation=45, labelsize=8)

        f.subplots_adjust(wspace=0.2, hspace=0.5)
        f.suptitle('All view plots', fontsize=20)
        plt.show()

        #Plot002 : ratio plots
        f = plt.figure()
        gs = gridspec.GridSpec(2, 2)

        ax1 = f.add_subplot(gs[0, 0])
        ax2 = f.add_subplot(gs[0, 1])
        ax3 = f.add_subplot(gs[1, :])

        ax1.plot(date, dislikeratio, color='red')
        ax1.set_title("Dislike / Total Like&Dis", fontsize=10)
        ax1.set_ylabel('Dislike ratio')
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45, labelsize=7)
        ax1.grid(which='major', linestyle='--')
        ax1.grid(which='minor', linestyle=':')

        ax2.plot(date, likeratio, color='blue')
        ax2.set_title("Like / Total Like&Dis", fontsize=10)
        ax2.set_ylabel('Like ratio')
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45, labelsize=7)
        ax2.grid(which='major', linestyle='--')
        ax2.grid(which='minor', linestyle=':')

        ax3.plot(date, fbratio)
        ax3.set_title("(Like+Dislike) / Total View", fontsize=10)
        ax3.set_ylabel('like&dis ratio')
        ax3.grid(True)
        ax3.tick_params(axis='x', rotation=45, labelsize=7)
        ax3.grid(which='major', linestyle='--')
        ax3.grid(which='minor', linestyle=':')

        f.subplots_adjust(wspace=0.2, hspace=0.5)
        f.suptitle('All ratio plots')
        plt.show()

    def seasonal_plot(self):
        sns.set()
        year = []
        month = []
        for i in range(0, len(date)):
            a = date[i].year
            year.append(a)
            b = date[i].month
            month.append(b)

        df = pd.DataFrame({'year': year, 'month': month, 'value': view})
        np.random.seed(100)
        mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(year), replace=False)

        plt.figure(figsize=(16, 12), dpi=80)
        for i, y in enumerate(year):
            if i > 0:
                plt.plot('month', 'value', data=df.loc[df.year == y, :], color=mycolors[i], label=y)
                plt.text(df.loc[df.year == y, :].shape[0] - .9, df.loc[df.year == y, 'value'][-1:].values[0], y,
                         fontsize=12, color=mycolors[i])

        plt.gca().set(ylabel='$view$', xlabel='$Month$')
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)
        plt.show()

    def anomaly(self, n_anomaly, type):
        #n_anomaly = 1, 2
        #type = 2d, 3d, all
        #select = view, like, dislike, comment
        sns.set()
        anomaly_df = pd.DataFrame({'date': date, 'title': title,'view': view, 'like': like, 'dislike': dislike, 'comment': comment})
        data = anomaly_df[['view', 'like', 'dislike', 'comment']]

        # IsolationForest Modelling
        clf = IsolationForest(n_estimators=100, max_samples='auto',
                              max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
        clf.fit(data)
        pred = clf.predict(data)
        anomaly_df['anomaly'] = pred

        # Outlier print
        outliers = anomaly_df.loc[anomaly_df['anomaly'] == -1]
        outlier_index = list(outliers.index)
        print("First anomaly selected values :")
        print(anomaly_df['anomaly'].value_counts())

        if type == '2d':
            # 2D Plot
            pca = PCA(2)
            pca.fit(data)
            res = pd.DataFrame(pca.transform(data))
            Z = np.array(res)
            figsize = (12, 7)
            plt.figure(figsize=figsize)
            plt.title("IsolationForest")
            plt.contourf(Z, cmap=plt.cm.Blues_r)

            b1 = plt.scatter(res[0], res[1], c='blue',
                             s=40, label="normal points")

            b1 = plt.scatter(res.iloc[outlier_index, 0], res.iloc[outlier_index, 1], c='red',
                             s=40, edgecolor="red", label="predicted outliers")
            plt.legend(loc="upper right")
            plt.title("2D highlighting anomalies")
            plt.show(block=True)

        elif type == '3d':
            # 3D Plot
            pca = PCA(n_components=3)
            scaler = StandardScaler()
            X = scaler.fit_transform(data)
            X_reduce = pca.fit_transform(X)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlabel("x_composite_3")
            ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers", c="green")
            ax.scatter(X_reduce[outlier_index, 0], X_reduce[outlier_index, 1], X_reduce[outlier_index, 2],
                       lw=2, s=60, marker="x", c="red", label="outliers")
            ax.legend()
            plt.title("3D highlighting anomalies")
            plt.show(block=True)
        else:
            # 2D Plot
            pca = PCA(2)
            pca.fit(data)
            res = pd.DataFrame(pca.transform(data))
            Z = np.array(res)
            figsize = (12, 7)
            plt.figure(figsize=figsize)
            plt.title("IsolationForest")
            plt.contourf(Z, cmap=plt.cm.Blues_r)

            b1 = plt.scatter(res[0], res[1], c='blue',
                             s=40, label="normal points")

            b1 = plt.scatter(res.iloc[outlier_index, 0], res.iloc[outlier_index, 1], c='red',
                             s=40, edgecolor="red", label="predicted outliers")
            plt.legend(loc="upper right")
            plt.title("2D highlighting anomalies")
            plt.show(block=True)

            # 3D Plot
            pca = PCA(n_components=3)
            scaler = StandardScaler()
            X = scaler.fit_transform(data)
            X_reduce = pca.fit_transform(X)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlabel("x_composite_3")
            ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers", c="green")
            ax.scatter(X_reduce[outlier_index, 0], X_reduce[outlier_index, 1], X_reduce[outlier_index, 2],
                       lw=2, s=60, marker="x", c="red", label="outliers")
            ax.legend()
            plt.title("3D highlighting anomalies")
            plt.show(block=True)

        # Plot anomaly
        fig, ax = plt.subplots(figsize=(10, 6))
        a = anomaly_df.loc[anomaly_df['anomaly'] == -1, ['date', 'view']]  # anomaly
        ax.plot(anomaly_df['date'], anomaly_df['view'], color='blue', label='view')
        ax.scatter(a['date'], a['view'], color='red', label='Anomaly')
        plt.legend()
        plt.title("First anomaly selected view counts")
        plt.show(block=True)

        if n_anomaly == 1:
            print(colored.fg('green'))
            print("MESSAGE : You selected '1' in no.of anomaly running.")
            print(colored.fg('white'))
            anomaly_only = anomaly_df.loc[anomaly_df['anomaly'] == -1]
        elif n_anomaly == 2:
            print(colored.fg('green'))
            print("MESSAGE : You selected '2' in no.of anomaly running.")
            print(colored.fg('white'))
            anomaly_only = anomaly_df.loc[anomaly_df['anomaly'] == -1]
            anomaly_only = anomaly_only.drop('anomaly', axis=1)  # Remove first anomaly selected
            del data
            data = anomaly_only[['view', 'like', 'dislike', 'comment']]
            clf = IsolationForest(n_estimators=100, max_samples='auto',
                                  max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
            clf.fit(data)
            pred = clf.predict(data)
            anomaly_only['anomaly'] = pred  # Add second anomaly selected

            print("Second anomaly selected values :")
            print(anomaly_only['anomaly'].value_counts())

            fig, ax = plt.subplots(figsize=(12, 8))
            a = anomaly_only.loc[anomaly_only['anomaly'] == -1, ['date', 'view']]  # anomaly
            ax.plot(anomaly_only['date'], anomaly_only['view'], color='blue', label='view')
            ax.scatter(a['date'], a['view'], color='red', label='Anomaly')
            plt.legend()
            plt.title("Second anomaly selected view counts")
            plt.show(block=True)

            # Select anomaly only
            anomaly_only = anomaly_only.loc[anomaly_only['anomaly'] == -1]
        else:
            print(colored.fg('green'))
            print("MESSAGE : You selected 'None' in no.of anomaly running. By deafault, your selection is 1")
            print(colored.fg('white'))
            anomaly_only = anomaly_df.loc[anomaly_df['anomaly'] == -1]

        # Tri plots for like and dislike
        f = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        ax1 = f.add_subplot(gs[0, 0])
        ax2 = f.add_subplot(gs[0, 1])
        ax3 = f.add_subplot(gs[1, :])

        ax1.plot(anomaly_only['date'], anomaly_only['dislike'], color='red')
        ax1.set_title("Dislike from anomaly selected", fontsize=10)
        ax1.set_ylabel('Dislike Count')
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45, labelsize=7)
        ax1.grid(which='major', linestyle='--')
        ax1.grid(which='minor', linestyle=':')

        ax2.plot(anomaly_only['date'], anomaly_only['like'], color='blue')
        ax2.set_title("Like from anomaly selected", fontsize=10)
        ax2.set_ylabel('Like Count')
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45, labelsize=7)
        ax2.grid(which='major', linestyle='--')
        ax2.grid(which='minor', linestyle=':')

        ax3.plot(anomaly_only['date'], anomaly_only['view'])
        ax3.set_title("View from anomaly selected", fontsize=10)
        ax3.set_ylabel('View Count')
        ax3.grid(True)
        ax3.tick_params(axis='x', rotation=45, labelsize=7)
        ax3.grid(which='major', linestyle='--')
        ax3.grid(which='minor', linestyle=':')

        f.subplots_adjust(wspace=0.2, hspace=0.5)
        f.suptitle('All plots in anomaly selected')

        # Like and Dislike in anomaly
        fig, ax = plt.subplots()
        ax.plot(anomaly_only['date'], anomaly_only['like'], label='Like')
        ax.plot(anomaly_only['date'], anomaly_only['dislike'], color='red', label='Dislike')
        ax.set_title("Like and dislike from anomaly selected", fontsize=18)
        ax.set_ylabel('Like & Dislike')
        ax.grid(True)
        plt.show()

        anomaly_date = anomaly_only[['date']]
        return anomaly_date

# Sentiment analysis
class sentiment(youtube):
    def __init__(self, api):
        super().__init__(api)

    def preprocessing(self, freq, rare):
        youtube = build('youtube', 'v3', developerKey=self.api)

        sns.set()
        print(colored.fg('green'))
        print('MESSAGE : Selected anomaly dates in your channel :', len(anomaly_date))
        print('MESSAGE : This is your date vide selected : ')
        print(colored.fg('white'))
        print(anomaly_date)

        # Step 1 : select anomaly date only in df
        df = pd.DataFrame(
            {'date': date, 'title': title, 'videoId': videoId, 'view': view, 'like': like, 'dislike': dislike,
             'comment': comment,
             'fbtotal': fbtotal, 'fbratio': fbratio, 'likeratio': likeratio, 'dislikeratio': dislikeratio})
        df = pd.merge(df, anomaly_date, on=['date'])
        df_videoid = df[['videoId']]

        # Step 2 : commentThreads()
        comment_df = []
        for i in range(0, len(df_videoid)):
            id = df_videoid['videoId'][i]
            print(i)
            res_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='relevance').execute()
            print(id)

            for j in range(0, len(res_comments['items'])):
                data = res_comments['items'][j]['snippet']['topLevelComment']['snippet']
                comment_df.append(data)

        # Step 3 : Extrect Text and numeric information
        comment_dic = []
        comment_sentiment = []
        for i in range(0, len(comment_df)):
            dic = {'videoId': comment_df[i]['videoId'], 'likeCount': comment_df[i]['likeCount'],
                   'publishedAt': comment_df[i]['publishedAt'], 'text': comment_df[i]['textOriginal']}
            comment_dic.append(dic)

            dic_title = {'publishedAt': comment_df[i]['publishedAt'], 'original_text': comment_df[i]['textOriginal']}
            comment_sentiment.append(dic_title)
            print(comment_sentiment[i])

        for i in range(0, len(comment_sentiment)):
            comment_sentiment[i]['publishedAt'] = datetime.strptime(comment_sentiment[i]['publishedAt'],
                                                                    '%Y-%m-%dT%H:%M:%SZ')
            comment_sentiment[i]['publishedAt'] = datetime.strftime(comment_sentiment[i]['publishedAt'], '%Y-%m-%d-%H')

        # Step 4 : Converting dataframe
        comment_sentiment = pd.DataFrame(comment_sentiment)

        # Step 5 : Text preprocessing
        # Ste 5 exists from part 001 to part 008
        ### Part 001 : Removal of Text
        comment_sentiment["transformed_text"] = comment_sentiment["original_text"].apply(lambda x: remove_cleantext(x))

        ### Part 002 : Removal of Emoji
        comment_sentiment["transformed_text"] = comment_sentiment["transformed_text"].apply(lambda text: remove_emoji(text))

        ### Part 003 : Removal of Emoticon
        comment_sentiment["transformed_text"] = comment_sentiment["transformed_text"].apply(lambda text: remove_emoticons(text))

        n_freq_words = freq
        n_rare_words = rare

        cnt = Counter()
        for text in comment_sentiment["transformed_text"].values:
            for word in text.split():
                cnt[word] += 1

        rem_frewords = set([w for (w, wc) in cnt.most_common(n_freq_words)])
        rem_rare = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])

        print(colored.fg('green'))
        print('MESSAGE : Frequent words :', rem_frewords)
        print('MESSAGE : Rare words :', rem_rare)
        print(colored.fg('white'))

        ### Part 004 : Removal of Freqwords
        def remove_freqwords(text):
            return " ".join([word for word in str(text).split() if word not in rem_frewords])
        comment_sentiment["transformed_text"] = comment_sentiment["transformed_text"].apply(lambda text: remove_freqwords(text))

        ### Part 005 : Removal of Rareword
        def remove_rarewords(text):
            return " ".join([word for word in str(text).split() if word not in rem_rare])
        comment_sentiment["transformed_text"] = comment_sentiment["transformed_text"].apply(lambda text: remove_rarewords(text))

        tokenizer = ToktokTokenizer()  # This tokenizer is for a better stopwords. Actual tokenization will be placed in sentiment score.
        rem_stopwords = set(stopwords.words('english'))

        ### Part 006 : Removal of Stopword
        def remove_stopwords(text):
            tokens = tokenizer.tokenize(text)
            tokens = [token.strip() for token in tokens]
            filtered_tokens = [token for token in tokens if token not in rem_stopwords]
            filtered_text = " ".join(filtered_tokens)
            return filtered_text
        comment_sentiment['transformed_text'] = comment_sentiment['transformed_text'].apply(remove_stopwords)

        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

        ### Part 007 : Removal of Stemming - Lemmatization (I am not using stemming : https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing)
        def lemmatize_words(text):
            pos_tagged_text = nltk.pos_tag(text.split())
            return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
        comment_sentiment['transformed_text'] = comment_sentiment['transformed_text'].apply(lambda text: lemmatize_words(text))

        ### Part 008 : Blacklist
        def tokenize(text):
            tokens = tokenizer.tokenize(text)
            return tokens
        comment_sentiment['earlytoken_text'] = comment_sentiment['transformed_text'].apply(tokenize)

        # Freq_word (Default 20)
        n_freq_words = 20
        count = Counter([item for sublist in comment_sentiment['earlytoken_text'] for item in sublist])
        show_count = pd.DataFrame(count.most_common(n_freq_words))
        show_count.columns = ['Common_words', 'count']
        show_count.style.background_gradient(cmap='Blues')
        print(show_count['Common_words'])

        # Figure
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.barh(show_count['Common_words'], show_count['count'])
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax.invert_yaxis()

        # Add annotation
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5, str(round((i.get_width()), 2)),
                     fontsize=10, fontweight='bold', color='grey')

        ax.set_title('Common words in every tokenizedw words Before BLACKLIST', loc='left', )
        fig.text(0.9, 0.15, 'Before Blacklist', fontsize=12, color='grey', ha='right', va='bottom', alpha=0.7)
        plt.show(block=True)

        # Set blacklist
        print("Please select your blacklist from the chart")
        blacklist = [item for item in input("Please type your blacklist. For example) Liverpool, Chelsea").split()]

        # Part 009 : Removal of your blacklist
        def remove_blacklist(text):
            return " ".join([word for word in str(text).split() if word not in blacklist])

        comment_sentiment["transformed_text"] = comment_sentiment["transformed_text"].apply(
            lambda text: remove_blacklist(text))
        comment_sentiment['earlytoken_text'] = comment_sentiment['transformed_text'].apply(tokenize)

        # Plot again
        n_freq_words = 20
        count = Counter([item for sublist in comment_sentiment['earlytoken_text'] for item in sublist])
        show_count = pd.DataFrame(count.most_common(n_freq_words))
        show_count.columns = ['Common_words', 'count']
        show_count.style.background_gradient(cmap='Blues')
        print(show_count['Common_words'])

        # Figure
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.barh(show_count['Common_words'], show_count['count'])
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax.invert_yaxis()

        # Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5, str(round((i.get_width()), 2)),
                     fontsize=10, fontweight='bold', color='grey')

        ax.set_title('Common words in every tokenizedw words After BLACKLIST', loc='left', )
        fig.text(0.9, 0.15, 'Before Blacklist', fontsize=12, color='grey', ha='right', va='bottom', alpha=0.7)
        plt.show(block=True)

        comment_sentiment.drop('earlytoken_text', axis=1, inplace=True)
        comment_sentiment.drop('original_text', axis=1, inplace=True)

        return comment_sentiment

    def sentiment_analysis(self):
        youtube = build('youtube', 'v3', developerKey=self.api)

        # Step 1 : Polarity and subjectivity
        comment_sentiment['polarity'] = comment_sentiment['transformed_text'].apply(lambda x: TextBlob(x).sentiment[0])
        comment_sentiment['subjectivity'] = comment_sentiment['transformed_text'].apply(lambda x: TextBlob(x).sentiment[0])

        sentiment_polarity = comment_sentiment.groupby(
            ['publishedAt'])['polarity'].apply(lambda x: x.astype(float).sum()).reset_index()

        sns.set()
        fig, ax = plt.subplots()
        ax.plot(sentiment_polarity['publishedAt'], sentiment_polarity['polarity'], label='Like')
        ax.set_title("Polarity in Youtue Comments", fontsize=18)
        ax.set_xlabel('Date')
        ax.set_ylabel('Polarity level')
        ax.grid(True)
        plt.show(block=True)

        # Step 2 : Pytorch sentiment analysis
        tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

        # Part 001 : Check Sequence Length
        token_lens = []
        for text in comment_sentiment.transformed_text:
            tokens = tokenizer.encode(text, truncation=True)
            token_lens.append(len(tokens))

        sns.displot(token_lens, kde=True).set(title='Tokenized counts and length')
        plt.xlim([0, 256])
        plt.xlabel('Token count')
        plt.show(block=True)

        # Part 002 : Check Sequence Length
        def sentiment_score(review):
            tokens = tokenizer.encode(review, return_tensors='pt')
            result = model(tokens)
            return int(torch.argmax(result.logits)) + 1

        comment_sentiment['sentiment'] = comment_sentiment['transformed_text'].apply(lambda x: sentiment_score(x[:250]))

        # Part 003 : Plot Sentiment Scores
        sentiment_bert = comment_sentiment.groupby(
            ['publishedAt'])['sentiment'].apply(lambda x: x.astype(float).sum()).reset_index()

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(sentiment_bert['publishedAt'], sentiment_bert['sentiment'], label='Like')
        ax.set_title("BERT sum of sentiment scores", fontsize=18)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sum of sentiment scores')
        ax.grid(True)
        plt.show(block=True)

        # Part 004 : Plot Sentiment Counts
        sentiment_count = comment_sentiment.groupby(
            ['sentiment'])['transformed_text'].agg(['nunique']).reset_index()

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.barh(sentiment_count['sentiment'], sentiment_count['nunique'])
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax.invert_yaxis()

        # Part 005 : Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5, str(round((i.get_width()), 2)),
                     fontsize=10, fontweight='bold', color='grey')

        ax.set_title('sentiment counts', loc='left', fontsize=18)
        plt.show(block=True)

        return comment_sentiment


################################################################################################################################################################################
######## TEXT TYPE :
api_credential = ""
channel_id = "UCYtNSrfGdXooZYu_hkq18_w"

######## VIDEO LIST SETUP/BASIC :
youtube = youtube(api=api_credential)

######## VIDEO LIST DETAIL :
trendvideo_list = youtube.trend_video(2021, 5, 21, after_day=7, n_max=10, region='MY', language='en')
popular_list = youtube.popular_video(n_max=20, region='MY')

######## CHANNEL ANALYSIS SETUP/BASIC :
summary = youtube.get_channel_stats(channel_id=channel_id, sort='date') #UCx6jsZ02B4K3SECUrkgPyzg
analysis = analysis(scaler=1)
fbtotal, fbratio, likeratio, dislikeratio = analysis.setup()

######## CHANNEL ANALYSIS DETAIL :
analysis.allchart()
analysis.seasonal_plot()
anomaly_date = analysis.anomaly(type='all', n_anomaly=2)

######## SENTIMENT ANALYSIS :
sentiment = sentiment(api=api_credential)
comment_sentiment = sentiment.preprocessing(freq=0, rare=10)
sentiment_analysis = sentiment.sentiment_analysis()