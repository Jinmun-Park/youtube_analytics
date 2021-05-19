# pip install google-api-python-client
from googleapiclient.discovery import build #https://developers.google.com/docs/api/quickstart/python
from datetime import datetime

#api_key = ""
youtube = build('youtube','v3',developerKey=api_key)
type(youtube)

# API Search list : https://developers.google.com/youtube/v3/docs/search/list
# PART 1
req_video = youtube.search().list(q='코로나',part='snippet', type='video', maxResults=50) # default = 5
res = req_video.execute()
#show_test = res['items'][0]['snippet']['title']
title_video = []
for item in res['items']:
    title_video.append(item['snippet']['title'])

# PART 2
req_channel = youtube.search().list(q='슈카월드',part='snippet', type='channel', maxResults=50, channelId='UCsJ6RuBiTVWRX156FVbeaGg')
res = req_channel.execute()
title_channel = []
for item in res['items']:
    title_channel.append(item['snippet']['title'])

# PART 3
start_time = datetime(year=2020, month=1, day=1).strftime('%Y-%m-%dT%H:%M:%SZ')
end_time = datetime(year=2021, month=1, day=1).strftime('%Y-%m-%dT%H:%M:%SZ')

req_video = youtube.search().list(q='코로나',part='snippet', type='video', maxResults=50, publishedAfter=start_time, publishedBefore=end_time)
res = req_video.execute()
for item in res['items']:
    print(item['snippet']['title'], item['snippet']['publishedAt'], item['id']['videoId'])
