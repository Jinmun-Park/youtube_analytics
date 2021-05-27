''':type
regionCode : http://www.loc.gov/standards/iso639-2/php/code_list.php
relevanceLanguage : https://www.iso.org/obp/ui/#search
google_api_searchlist : https://developers.google.com/youtube/v3/docs/search/list
'''
# pip install google-api-python-client
from googleapiclient.discovery import build #API
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl # Seasonal plot
import matplotlib.gridspec as gridspec #Splitplot
import seaborn as sns #Graph Visualization
import colored
from sklearn.decomposition import PCA #Anomaly
from sklearn.preprocessing import StandardScaler #Anomaly
from sklearn.ensemble import IsolationForest #Anomaly

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
            extra = str(summary[i]['title'])
            title.append(extra)

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

    def anomaly_plot(self, n_anomaly, type):
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
            print("You selected '1' in no.of anomaly running.")
            anomaly_only = anomaly_df.loc[anomaly_df['anomaly'] == -1]
        elif n_anomaly == 2:
            print("You selected '2' in no.of anomaly running.")
            anomaly_only = anomaly_df.loc[anomaly_df['anomaly'] == -1]
            anomaly_only = anomaly_only.drop('anomaly', axis=1)  # Remove first anomaly selected
            del data
            data = anomaly_only[['view', 'like', 'dislike']]
            clf = IsolationForest(n_estimators=100, max_samples='auto',
                                  max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
            clf.fit(data)
            pred = clf.predict(data)
            anomaly_only['anomaly'] = pred  # Add second anomaly selected

            outliers = anomaly_only.loc[anomaly_only['anomaly'] == -1]
            outlier_index = list(outliers.index)
            print("Second anomaly selected values :")
            print(anomaly_only['anomaly'].value_counts())

            fig, ax = plt.subplots(figsize=(10, 6))
            a = anomaly_only.loc[anomaly_only['anomaly'] == -1, ['date', 'view']]  # anomaly
            ax.plot(anomaly_only['date'], anomaly_only['view'], color='blue', label='view')
            ax.scatter(a['date'], a['view'], color='red', label='Anomaly')
            plt.legend()
            plt.title("Second anomaly selected view counts")
            plt.show(block=True)

        else:
            print("You selected 'None' in no.of anomaly running. By deafault, your selection is 1")
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

################################################################################################################################################################################
######## VIDEO LIST :
youtube = youtube("AIzaSyAM1a_XGQnnLDyJ7oYmhJV8mBDRY7MDtxk")
trendvideo_list = youtube.trend_video(2021, 5, 19, after_day=7, n_max=10)
popular_list = youtube.popular_video(n_max=20)

######## CHANNEL ANALYSIS :
summary = youtube.get_channel_stats('UCx6jsZ02B4K3SECUrkgPyzg', sort='date')
analysis = analysis(scaler=1)
fbtotal, fbratio, likeratio, dislikeratio = analysis.setup()
analysis.allchart()
analysis.seasonal_plot()
analysis.anomaly_plot(type='all', n_anomaly=2)

################################################################################################################################################################################
######## TEST PLACE :
youtube = build('youtube', 'v3', developerKey="AIzaSyAM1a_XGQnnLDyJ7oYmhJV8mBDRY7MDtxk")
channel_id = 'UCx6jsZ02B4K3SECUrkgPyzg'
# anomaly detection
# https://towardsdatascience.com/time-series-of-price-anomaly-detection-13586cd5ff46
# https://neptune.ai/blog/anomaly-detection-in-time-series#:~:text=What%20are%20anomalies%2Foutliers%20and,generated%20by%20a%20different%20mechanism.%E2%80%9D

# add trend line
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs