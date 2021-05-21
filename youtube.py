''':type
regionCode : http://www.loc.gov/standards/iso639-2/php/code_list.php
relevanceLanguage : https://www.iso.org/obp/ui/#search
'''
# pip install google-api-python-client
from googleapiclient.discovery import build #https://developers.google.com/docs/api/quickstart/python
from datetime import datetime
import colored

class youtube:
    def __init__(self, api):
        self.api = api

    def dailytrend(self, year, month, day, n_max):

        youtube = build('youtube', 'v3', developerKey=self.api)
        print(colored.fg('green'))
        print("MESSAGE : Your start year/month/day :", year, "/", month, "/", day)
        print("MESSAGE : Your end year/month/day :", year, "/", month, "/", day+1)
        print(colored.fg('white'))
        start_time = datetime(year=year, month=month, day=day).strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = datetime(year=year, month=month, day=day+1).strftime('%Y-%m-%dT%H:%M:%SZ')

        res_daytrend = youtube.search().list(part='snippet', type='video', order='viewCount', maxResults=n_max,
                                             regionCode='KR', relevanceLanguage='ko', publishedAfter=start_time,
                                             publishedBefore=end_time).execute()
        print(colored.fg('green'))
        print("MESSAGE : List of channel,video titles and channel id in top view counts")
        print(colored.fg('white'))

        dailytrend_list = []

        for index, item in enumerate(res_daytrend['items']): # enumerate can show a sequence of loop
            dic = {'channelTitle':item['snippet']['channelTitle'], 'Video Title':item['snippet']['title'], 'ChannelID':item['snippet']['channelId']}
            dailytrend_list.append(dic)
            print(index, "|Channel Title :|", item['snippet']['channelTitle'], "\n", index, "|Video Title :|", item['snippet']['title'], "\n", index, "|ChannelID :|", item['snippet']['channelId'])

        return dailytrend_list

youtube = youtube("AIzaSyAM1a_XGQnnLDyJ7oYmhJV8mBDRY7MDtxk")
youtube.dailytrend(2021, 5, 19, n_max=10)

#api_key = ""
youtube = build('youtube','v3',developerKey=api_key)

# API Search list : https://developers.google.com/youtube/v3/docs/search/list
# PART 1 : Find Video
start_time = datetime(year=2021, month=5, day=19).strftime('%Y-%m-%dT%H:%M:%SZ')
end_time = datetime(year=2021, month=5, day=20).strftime('%Y-%m-%dT%H:%M:%SZ')
req_video = youtube.search().list(q='슈카월드',part='snippet', type='video', maxResults=50, publishedAfter=start_time, publishedBefore=end_time)
res = req_video.execute()

test_res = youtube.search().list(part='snippet', type='video', order='viewCount', maxResults=5, regionCode='KR', relevanceLanguage='ko',publishedAfter=start_time, publishedBefore=end_time).execute()
count = 0
for count, item in enumerate(test_res['items']):
    print(count + 1)
    print("|Channel Title :|", item['snippet']['channelTitle'], "|Video Title :|", item['snippet']['title'], "|ChannelID :|", item['snippet']['channelId'])

dic_list = []
for item in sorted(res['items'], key=lambda x:x['snippet']['publishedAt']):
    dic = {'key1':item['snippet']['title'], 'key2':item['snippet']['channelTitle'], 'key3':item['snippet']['channelId'], 'key4':item['id']['videoId']}
    print(item['snippet']['title'], item['snippet']['channelTitle'], item['snippet']['channelId'], item['id']['videoId'])
    dic_list.append(dic)

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
req_playlistitems = youtube.playlistItems().list(part='snippet', playlistId='UUsJ6RuBiTVWRX156FVbeaGg', maxResults=50)
res_playlistitems = req_playlistitems.execute() #totalResults = 829

#res_channel = youtube.channels().list(part='snippet', id=channel_id).execute() #Show country ['items']['snippet']['country']

def get_channel_videos(channel_id):
    res_statistic = youtube.channels().list(part='statistics', id=channel_id).execute()
    print("* SubscriberCount :", res_statistic['items'][0]['statistics']['subscriberCount'])
    print("* VideoCount :", res_statistic['items'][0]['statistics']['videoCount'])

    res_channel = youtube.channels().list(part='contentDetails', id=channel_id).execute()
    print("* Playlist ID :", res_channel['items'][0]['contentDetails']['relatedPlaylists']['uploads'])
    playlist_id = res_channel['items'][0]['contentDetails']['relatedPlaylists']['uploads'] #Export Playlist ID. Playlist ID is different from Channel ID.

    videos = []
    next_page_token = None

    while 1:
        res = youtube.playlistItems().list(part='snippet', playlistId=playlist_id, maxResults=50, pageToken=next_page_token).execute()
        videos+=res['items']
        next_page_token = res.get('nextPageToken')
        if next_page_token is None:
            break
    print("* Playlist Result Counts :", res['pageInfo']['totalResults'])
    return videos

videos = get_channel_videos('UCsJ6RuBiTVWRX156FVbeaGg')
len(videos)

#for video in sorted(videos, key=lambda x:x['snippet']['publishedAt']):
#    print(video['snippet']['title'],video['snippet']['publishedAt'])

# PART 4 : Statistic
video_ids = list(map(lambda x:x['snippet']['resourceId']['videoId'], videos))

def get_videos_stats(video_ids):
    stats = []
    rating = []
    title = []
    summary = []

    for i in range(0, len(video_ids), 50): #Sequence : 0~50, 50~100,,,
        res_stat = youtube.videos().list(id=','.join(video_ids[i:i+50]), part='statistics').execute()
        res_rate = youtube.videos().list(id=','.join(video_ids[i:i+50]), part='contentDetails').execute()
        res_title = youtube.videos().list(id=','.join(video_ids[i:i + 50]), part='snippet').execute()
        stats += res_stat['items']
        rating += res_rate['items']
        title += res_title['items']

    for i in range(0, len(stats)):
        dic = {'title':title[i]['snippet']['title'], 'videoId':stats[i]['id'], 'statistics': stats[i]['statistics'], 'contentRating': rating[i]['contentDetails']['contentRating']}
        summary.append(dic)
    return summary, stats, rating, title

summary, stats, rating, title = get_videos_stats(video_ids)

most_disliked = sorted(stats, key=lambda x:int(x['statistics']['dislikeCount']), reverse=True)
most_disliked
