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

# Part 001 : clean text
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

# Part 002 : Removal of Emoji & Emoticons
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

# Part 003 : Removal of Emoticons
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

# Part 004 : Removal of Freqwords
# Part 005 : Removal of Rarewords
# Part 006 : Removal of Stopwords
# Part 007 : Lemmatization (I am not using stemming : https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing)

# Part 008 : Tokenization for blacklist
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return tokens

# Part 009 : Removal of your blacklist
def remove_blacklist(text):
    return " ".join([word for word in str(text).split() if word not in blacklist])

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

        # Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5, str(round((i.get_width()), 2)),
                     fontsize=10, fontweight='bold', color='grey')

        ax.set_title('Common words in every tokenizedw words Before BLACKLIS', loc='left', )
        fig.text(0.9, 0.15, 'Before Blacklist', fontsize=12, color='grey', ha='right', va='bottom', alpha=0.7)
        plt.show(block=True)

        # T006-002 : Set blacklist
        blacklist = ["’"]

        def remove_blacklist(text):
            return " ".join([word for word in str(text).split() if word not in blacklist])

        comment_sentiment["transformed_text"] = comment_sentiment["transformed_text"].apply(
            lambda text: remove_blacklist(text))
        comment_sentiment['earlytoken_text'] = comment_sentiment['transformed_text'].apply(tokenize)

        # T006-003 : Plot again
        n_freq_words = 20
        count = Counter([item for sublist in comment_sentiment['earlytoken_text'] for item in sublist])
        show_count = pd.DataFrame(count.most_common(n_freq_words))
        show_count.columns = ['Common_words', 'count']
        show_count.style.background_gradient(cmap='Blues')

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

        ax.set_title('Common words in every tokenizedw words After BLACKLIS', loc='left', )
        fig.text(0.9, 0.15, 'Before Blacklist', fontsize=12, color='grey', ha='right', va='bottom', alpha=0.7)
        plt.show(block=True)

        comment_sentiment.drop('earlytoken_text', axis=1, inplace=True)
        comment_sentiment.drop('original_text', axis=1, inplace=True)

        return comment_sentiment


######## SENTIMENT ANALYSIS :
sentiment = sentiment(api=api_credential)
sentiment = sentiment.preprocessing(freq=0, rare=10)

test = [input("Type your blacklist. Example) Chelsea, Liverpool")]
