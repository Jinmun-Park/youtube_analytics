# youtube_analytics

## Project Overview
<p align="justify">This project is to performa sentiment analysis using Google API. The objective in this project is to extract numeric information such as like/dislike/comment counts/views and perform sentiment analysis with the comments in your selected videos. One of the goals is to build class function enables us to have a list of trend videos in your selected region and we extract the channel ID. We use this channel ID to extract all available video IDs and pull comments to perform sentiment analysis. </p>

The final goal is to create web app using flask or docker for end-users to have their analysis using few mouse clicks. 

***

## Youtube API
You are required to request for GoogleAPI at initial stage. Please follow the link below to have your API.
1. Please apply your google api in the URL below : [GoogleAPI][GoogleAPI]
2. A list of details in Google API can be found here : [GoogleAPI_reference][GoogleAPI_reference]

<p align="justify">Please watch youtube channel ClarityCoders to request API If you have trouble in getting API key. I get alot of help understanding from his video.</p>

This is his youtube video : [ClarityCoders][ClarityCoders].

***

## How to use it
I have designed function in three main parts : Getting channel ID, video dates are selected using anomaly detection and sentiment analysis.

### Youtube API and find out your channel
Please use `youtube` function to have a list of trend videos. 
- `youtube.trendvideo(year, month, day, after_day, n_max, region, language)` This uses **search list** option to extract trend channels using date,region and language filters. 
- `popular_video(n_max, region)` This uses **video list** option to extract the mostPopular videos using region filter.
- `get_channel_stats(channel_id, sort)` Please keyin your **ChannelID** to get numerif figures in the channel.

<p align="justify">This will give a list of channel ID and we will pick up a single channel to have a deeper analysis. Please have your region/lanaguage unicode using link below ; </p>

1. [regionCode][regionCode] 
2. [relevanceLanguage][relevanceLanguage] 

### Perform simple EDA on your selected youtube channel

Please use `analysis` function to have visualization plots in your selected channel. 

```
summary = youtube.get_channel_stats('YourAPI', sort='date') 
analysis = analysis(scaler=1)
fbtotal, fbratio, likeratio, dislikeratio = analysis.setup()
analysis.allchart()
analysis.seasonal_plot()
anomaly_date = analysis.anomaly(type='all', n_anomaly=2)
```

### Perform Sentiment analysis using Textblob and BERT
```
sentiment = sentiment(api=api_credential)
comment_sentiment = sentiment.preprocessing(freq=0, rare=10)
sentiment_analysis = sentiment.sentiment_analysis()
```



[GoogleAPI]:https://console.cloud.google.com/cloud-resource-manager
[GoogleAPI_reference]:https://developers.google.com/youtube/v3/docs/?apix=true
[ClarityCoders]:https://www.youtube.com/watch?v=2mSwcRb3KjQ&t=1030s&ab_channel=ClarityCodersClarityCoders
[regionCode]: https://www.iso.org/iso-3166-country-codes.html
[relevanceLanguage]: https://www.loc.gov/standards/iso639-2/php/code_list.php
