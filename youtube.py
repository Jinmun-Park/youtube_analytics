''':type
regionCode : http://www.loc.gov/standards/iso639-2/php/code_list.php
relevanceLanguage : https://www.iso.org/obp/ui/#search
google_api_searchlist : https://developers.google.com/youtube/v3/docs/search/list
'''
# pip install google-api-python-client
from googleapiclient.discovery import build
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import colored

# API Search list :
class youtube:
    def __init__(self, api):
        self.api = api

    def trend_video(self, year, month, day, after_day, n_max):

        youtube = build('youtube', 'v3', developerKey=self.api)
        print(colored.fg('green'))
        print("MESSAGE : Your start year/month/day :", year, "/", month, "/", day)
        print("MESSAGE : Your end year/month/day :", year, "/", month, "/", day+after_day)
        print(colored.fg('white'))
        start_time = datetime(year=year, month=month, day=day).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = datetime(year=year, month=month, day=day+after_day).strftime('%Y-%m-%dT%H:%M:%SZ')

        res_trendvideo = youtube.search().list(part='snippet', type='video', order='viewCount', maxResults=n_max,
                                             regionCode='KR', relevanceLanguage='ko', publishedAfter=start_time,
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

    def popular_video(self, n_max):

        youtube = build('youtube', 'v3', developerKey=self.api)
        print(colored.fg('green'))
        print("MESSAGE : Your selected list of popular videos are :", n_max)
        print(colored.fg('white'))

        res_popular = youtube.videos().list(part='snippet', chart='mostPopular',
                                            maxResults=n_max, regionCode='KR').execute()

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

class analysis:
    def __init__(self, scaler):
        self.scaler = int(scaler)
    #def cursor_chart(self):

    def allchart(self):
        # All plots list
        date = []
        view = []
        like = []
        dislike = []
        comment = []
        #  ratio plots list
        fbtotal = [] #like+dislike
        fbratio = [] #fbtotal/view
        likeratio = [] #like/fbtotal
        dislikeratio = [] #dislike/fbtotal

        for i in range(0, len(summary)):
            # All plots
            x = summary[i]['publishedAt']
            date.append(x)
            a = int(summary[i]['statistics']['viewCount'])/self.scaler
            view.append(a)
            b = int(summary[i]['statistics']['likeCount'])/self.scaler
            like.append(b)
            c = int(summary[i]['statistics']['dislikeCount'])/self.scaler
            dislike.append(c)
            d = int(summary[i]['statistics']['commentCount'])/self.scaler
            comment.append(d)

        for i in range(0, len(summary)):
            # Ratio plots
            e = like[i]+dislike[i] #like+dislike
            fbtotal.append(e)

        for i in range(0, len(summary)):
            f = fbtotal[i]/view[i] #fbtotal/view
            fbratio.append(f)
            g =  like[i]/fbtotal[i] #like/fbtotal
            likeratio.append(g)
            h = dislike[i]/fbtotal[i] #dislike/fbtotal
            dislikeratio.append(h)

        # Plot_001 : All plots
        sns.set()
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 14), dpi=80)
        ax1.plot(date, view, label='ViewCount')
        ax1.set_title("ViewCount", fontsize=15)
        ax1.set_ylabel('ViewCount')
        ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)

        ax2.plot(date, like, color='red', label='LikeCount')
        ax2.set_title("LikeCount", fontsize=15)
        ax2.set_ylabel('LikeCount')
        ax2.grid(True)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)

        ax3.plot(date, dislike, color='blue', label='DislikeCount')
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
        plt.show(block=True)

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

        ax2.plot(date, likeratio, color='black')
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
        plt.show(block=True)


################################################################################################################################################################################
youtube = youtube("AIzaSyAM1a_XGQnnLDyJ7oYmhJV8mBDRY7MDtxk")
trendvideo_list = youtube.trend_video(2021, 5, 19, after_day=7, n_max=10)
popular_list = youtube.popular_video(n_max=20)
summary = youtube.get_channel_stats('UCPKNKldggioffXPkSmjs5lQ', sort='date')

analysis = analysis(scaler=1)
analysis.allchart()

youtube = build('youtube', 'v3', developerKey="") #AIzaSyAM1a_XGQnnLDyJ7oYmhJV8mBDRY7MDtxk
################################################################################################################################################################################
######## TEST PLACE :

date = []
view = []
like = []
dislike = []
comment = []
#  ratio plots list
fbtotal = []  # like+dislike
fbratio = []  # fbtotal/view
likeratio = []  # like/fbtotal
dislikeratio = []  # dislike/fbtotal

for i in range(0, len(summary)):
    # All plots
    x = summary[i]['publishedAt']
    date.append(x)
    a = int(summary[i]['statistics']['viewCount'])
    view.append(a)
    b = int(summary[i]['statistics']['likeCount'])
    like.append(b)
    c = int(summary[i]['statistics']['dislikeCount'])
    dislike.append(c)
    d = int(summary[i]['statistics']['commentCount'])
    comment.append(d)

for i in range(0, len(summary)):
    # Ratio plots
    e = like[i] + dislike[i]  # like+dislike
    fbtotal.append(e)

for i in range(0, len(summary)):
    f = fbtotal[i] / view[i]  # fbtotal/view
    fbratio.append(f)
    g = like[i] / fbtotal[i]  # like/fbtotal
    likeratio.append(g)
    h = dislike[i] / fbtotal[i]  # dislike/fbtotal
    dislikeratio.append(h)



