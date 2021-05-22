''':type
regionCode : http://www.loc.gov/standards/iso639-2/php/code_list.php
relevanceLanguage : https://www.iso.org/obp/ui/#search
'''
# pip install google-api-python-client
from googleapiclient.discovery import build #https://developers.google.com/docs/api/quickstart/python
from datetime import datetime
import colored

# API Search list : https://developers.google.com/youtube/v3/docs/search/list
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
            dic = {'channelTitle':item['snippet']['channelTitle'], 'Video Title':item['snippet']['title'], 'ChannelID':item['snippet']['channelId']}
            trendvideo_list.append(dic)
            print(index, "|Channel Title :|", item['snippet']['channelTitle'], "\n", index, "|Video Title :|", item['snippet']['title'], "\n", index, "|ChannelID :|", item['snippet']['channelId'])

        return trendvideo_list

    def popular_video(self, n_max):

        youtube = build('youtube', 'v3', developerKey=self.api)
        print(colored.fg('green'))
        print("MESSAGE : Your selected list of popular videos are :", n_max)
        print(colored.fg('white'))

        res_popular = youtube.videos().list(part='snippet', chart='mostPopular', maxResults=n_max, regionCode='KR').execute()

        print(colored.fg('green'))
        print("MESSAGE : List of channel,video titles and channel id in top view counts")
        print(colored.fg('white'))

        popular_list = []

        for index, item in enumerate(res_popular['items']): # enumerate can show a sequence of loop
            dic = {'channelTitle':item['snippet']['channelTitle'], 'Video Title':item['snippet']['title'], 'ChannelID':item['snippet']['channelId'], 'publishedAt':item['snippet']['publishedAt']}
            popular_list.append(dic)
            print(index, "|Channel Title :|", item['snippet']['channelTitle'], "\n",
                  index, "|Video Title :|", item['snippet']['title'], "\n",
                  index, "|ChannelID :|", item['snippet']['channelId'], "\n",
                  index, "|publishedAt :|", item['snippet']['publishedAt'])

        return popular_list

    def get_channel_videos(self, channel_id, n_max):

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
            res = youtube.playlistItems().list(part='snippet', playlistId=playlist_id, maxResults=n_max,
                                               pageToken=next_page_token).execute()
            videos += res['items']
            next_page_token = res.get('nextPageToken')
            if next_page_token is None:
                break
        print("MESSAGE : Playlist Result Counts :", res['pageInfo']['totalResults'])
        print(colored.fg('white'))

        return videos

    def get_videos_stats(self, video_ids):
        print(colored.fg('green'))
        print("INPUT : Please write text to set your sort conditions")
        print(colored.fg('white'))
        sort = input("Conditions are : date, view, like, dislike :")
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
        else:
            summary = sorted(summary, key=lambda x: x['publishedAt'], reverse=True)
        return summary
################################################################################################################################################################################
youtube = youtube() #"AIzaSyAM1a_XGQnnLDyJ7oYmhJV8mBDRY7MDtxk"
trendvideo_list = youtube.trend_video(2021, 5, 19, after_day=7, n_max=10)
popular_list = youtube.popular_video(n_max=10)
videos = youtube.get_channel_videos('', n_max = 50) # UCDV9zgWo7b6nPg7i49oRQ5Q

video_ids = list(map(lambda x:x['snippet']['resourceId']['videoId'], videos))
summary = youtube.get_videos_stats(video_ids)

youtube = build('youtube','v3',developerKey="") #AIzaSyAM1a_XGQnnLDyJ7oYmhJV8mBDRY7MDtxk
################################################################################################################################################################################
######## TEST PLACE : Have to change the
res_statistic = youtube.channels().list(part='snippet', id='').execute() # UCDV9zgWo7b6nPg7i49oRQ5Q
start_time = datetime(2021, 1, 1).strftime('%Y-%m-%dT%H:%M:%SZ')
convert_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%SZ')


# PART 2 : Find Channel Name
res_search = youtube.search().list(q='슈카월드', part='snippet', type='channel').execute()
for item in res_search['items']:
    print(item['snippet']['channelTitle'], '***'*3, item['id']['channelId'])

# PART 3
'''
# Videocounts : The number of public videos uploaded to the channel. Note that the value reflects the count of the channel's public videos only
req_channel = youtube.channels().list(part='statistics', id='UCsJ6RuBiTVWRX156FVbeaGg')
res_channel = req_channel.execute()

#contentDetails.uploads : The ID of the playlist that contains the channel's uploaded videos.
req_channel = youtube.channels().list(part='contentDetails', id='UCsJ6RuBiTVWRX156FVbeaGg')
res_channel = req_channel.execute() #playlist_id='UUsJ6RuBiTVWRX156FVbeaGg'
print(res_channel['items'][0]['contentDetails']['relatedPlaylists']['uploads'])
'''
# playlistItems() : video, that is included in a playlist
req_playlistitems = youtube.playlistItems().list(part='snippet', playlistId='', maxResults=50) #UUsJ6RuBiTVWRX156FVbeaGg
res_playlistitems = req_playlistitems.execute() #totalResults = 829

# PART 4 : Statistic