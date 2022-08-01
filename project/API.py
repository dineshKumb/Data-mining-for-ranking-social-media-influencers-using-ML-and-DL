import streamlit as st
import login
import pandas as pd
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import plotly.express as px
import datetime
from datetime import timedelta
from datetime import datetime
import xgboost
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,roc_auc_score,\
                            r2_score, mean_squared_error, recall_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from textblob import TextBlob

import sqlite3
conn=sqlite3.connect('data.db')
c=conn.cursor()

api_key='AIzaSyDjF6w3QnVjpYhMjA6oIy3pf-12BKbMRgI'
channel_ids=[channelID]
youtube=build('youtube','v3', developerKey=api_key)

def get_channel_stats(youtube, channel_ids): 
    request=youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=','.join(channel_ids))
    response= request.execute()
    all_data=[]
    for i in range(len(response['items'])):
        data = dict(Channel_name=response['items'][i]['snippet']['title'],
                Description=response['items'][i]['snippet']['description'],
                Country=response['items'][i]['snippet']['country'],
                Subscribers=response['items'][i]['statistics']['subscriberCount'],
                Views=response['items'][i]['statistics']['viewCount'],
                Total_videos=response['items'][i]['statistics']['videoCount'],
                playlist_id=response['items'][i]['contentDetails']['relatedPlaylists']['uploads'])
                #country_name=response['item'][i][]
        all_data.append(data)  
        return all_data
get_channel_stats(youtube, channel_ids)
channel_satistics=get_channel_stats(youtube, channel_ids)
channel_data=pd.DataFrame(channel_satistics)

playlist_id = channel_data.loc[channel_data['Channel_name']==channel_data['Channel_name'], 'playlist_id'].iloc[0]

channel_data['Subscribers']=pd.to_numeric(channel_data['Subscribers'])
channel_data['Views']=pd.to_numeric(channel_data['Views'])
channel_data['Total_videos']=pd.to_numeric(channel_data['Total_videos'])

def get_video_ids(youtube, playlist_id):
    
    request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId = playlist_id,
                maxResults = 50)
    response = request.execute()
    
    video_ids = []
    
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    more_pages = True
    
    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                        part='contentDetails',
                        playlistId = playlist_id,
                        maxResults = 50,
                        pageToken = next_page_token)
            response = request.execute()
    
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
        
    return video_ids

video_ids1=get_video_ids(youtube, playlist_id)

def get_video_details(youtube, video_ids1):
    all_video_stats=[]
    
    for i in range(0,len(video_ids1), 50):
        request=youtube.videos().list(
            part='snippet,statistics',
            id=','.join(video_ids1[i:i+50]))
        response=request.execute()
    
        for video in response['items']:
            
            video_stats=dict(Title=video['snippet']['title'],
                        Published_date=video['snippet']['publishedAt'],
                        Views=video['statistics']['viewCount'],
                        Likes=video['statistics']['likeCount'],
                        Comments = video['statistics']['commentCount'],
                        Description=video['snippet']['description'],
                        #Country=video['snippet']['country']
                        )
            all_video_stats.append(video_stats)

    return all_video_stats

video_data=get_video_details(youtube, video_ids1)
video_details=pd.DataFrame(video_data)

video_details['Published_date']=pd.to_datetime(video_details['Published_date']).dt.date
video_details['Views']=pd.to_numeric(video_details['Views'])
video_details['Likes']=pd.to_numeric(video_details['Likes'])
video_details['Comments']=pd.to_numeric(video_details['Comments'])
video_details_filtered=video_details.drop_duplicates()

video_details_filtered['Video_Id']=video_ids1


#all in one graph for top 10 videos based on views
st.subheader('Top 10 videos based on Views')
top10_videos=video_details_filtered.sort_values(by='Views', ascending=False).head(10)
st.dataframe(top10_videos)
groupby_column=st.selectbox(
    'What you like to analyse?',
    ('Likes','Comments','Title'))
output_columns=['Views']
df_grouped=top10_videos.groupby(by=[groupby_column],as_index=False)[output_columns].sum()
# st.dataframe(df_grouped)
fig=px.line(
    df_grouped,
    x=groupby_column,
    y='Views',
    text=groupby_column,
    color=groupby_column)
st.plotly_chart(fig)


#all in one graph for top 10 videos based on Likes
st.subheader('Top 10 videos based on Likes')
top10_videos=video_details_filtered.sort_values(by='Likes', ascending=False).head(10)
st.dataframe(top10_videos)
groupby_column=st.selectbox(
    'What you like to analyse?',
    ('Views','Comments','Title'))
output_columns=['Likes']
df_grouped=top10_videos.groupby(by=[groupby_column],as_index=False)[output_columns].sum()
# st.dataframe(df_grouped)
fig=px.line(
    df_grouped,
    x=groupby_column,
    y='Likes',
    text=groupby_column,
    color=groupby_column)
st.plotly_chart(fig)


#all in one graph for top 10 videos based on comments
st.subheader('Top 10 videos based on Comments')
top10_videos=video_details_filtered.sort_values(by='Comments', ascending=False).head(10)
st.dataframe(top10_videos)
groupby_column=st.selectbox(
    'What you like to analyse?',
    ('Likes','Views','Title'))
output_columns=['Comments']
df_grouped=top10_videos.groupby(by=[groupby_column],as_index=False)[output_columns].sum()
# st.dataframe(df_grouped)
fig=px.line(
    df_grouped,
    x=groupby_column,
    y='Comments',
    text=groupby_column,
    color=groupby_column)
st.plotly_chart(fig)


st.subheader('Yearly Analysis')
st.write("Views vs Years")
#yearly views
video_details_filtered[["year", "month", "day"]] = video_details_filtered["Published_date"].astype(str).str.split("-", expand = True)
year=video_details_filtered.year.unique().tolist()
output_columns=['Views']
df_grouped=video_details_filtered.groupby(by='year',as_index=False)['Views'].sum()
# st.dataframe(df_grouped)
fig=px.bar(
    df_grouped,
    x='year',
    y='Views',
    text='Views',
    color=year)
st.plotly_chart(fig)

st.write("Likes vs Years")
#yearly likes
output_columns=['Likes']
df_grouped=video_details_filtered.groupby(by='year',as_index=False)['Likes'].sum()
# st.dataframe(df_grouped)
fig=px.bar(
    df_grouped,
    x='year',
    y='Likes',
    text='Likes',
    color=year)
st.plotly_chart(fig)


#yearly comments
st.write("Comments vs Years")
output_columns=['Comments']
df_grouped=video_details_filtered.groupby(by='year',as_index=False)['Comments'].sum()
# st.dataframe(df_grouped)
fig=px.bar(
    df_grouped,
    x='year',
    y='Comments',
    text='Comments',
    color=year)
st.plotly_chart(fig)


from keybert import KeyBERT
title=[]
for i in video_details_filtered.Title:
    title.append(i)
Title=" ".join(map(str,title))
model=KeyBERT('distilbert-base-nli-mean-tokens')
keywords=model.extract_keywords(Title)

st.subheader("Latest Video Topics Are As Follows")
for i in keywords:
    st.write(i[0])

sortedbycomment=video_details_filtered.sort_values('Comments')

sortedcomm=sortedbycomment[:5]
fifty_comments_videos=sortedcomm['Video_Id']

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#video_ids2=video_ids1[:50]    
# # Replace this YouTube video ID with your own.
box = [['Comment', 'Likes', 'Reply Count']]

def scrape_comments_with_replies(fifty_comments_videos):
        for ID in fifty_comments_videos:
            data = youtube.commentThreads().list(part='snippet', videoId=ID, maxResults='100', textFormat="plainText").execute()

            for i in data["items"]:
                comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
                likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
                replies = i["snippet"]['totalReplyCount']

                box.append([comment, likes, replies])

                totalReplyCount = i["snippet"]['totalReplyCount']

                if totalReplyCount > 0:

                    parent = i["snippet"]['topLevelComment']["id"]

                    data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                    textFormat="plainText").execute()

                    for i in data2["items"]:
                        comment = i["snippet"]["textDisplay"]
                        likes = i["snippet"]['likeCount']
                        replies = ""

                        box.append([comment, likes, replies])

            while ("nextPageToken" in data):

                data = youtube.commentThreads().list(part='snippet', videoId=ID, pageToken=data["nextPageToken"],
                                                     maxResults='100', textFormat="plainText").execute()

                for i in data["items"]:
                    comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
                    likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
                    replies = i["snippet"]['totalReplyCount']

                    box.append([comment, likes, replies])

                    totalReplyCount = i["snippet"]['totalReplyCount']

                    if totalReplyCount > 0:

                        parent = i["snippet"]['topLevelComment']["id"]

                        data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                        textFormat="plainText").execute()

                        for i in data2["items"]:
                            comment = i["snippet"]["textDisplay"]
                            likes = i["snippet"]['likeCount']
                            replies = ''

                            box.append([comment, likes, replies])

            df = pd.DataFrame({'Comment': [i[0] for i in box],'Likes': [i[1] for i in box], 'Reply Count': [i[2] for i in box]})

        #df.to_
        ('youtube-comments.csv', index=False, header=False)
        return df


df_comments=scrape_comments_with_replies(fifty_comments_videos)

video_dataset=video_details_filtered
video_dataset[["year", "month", "day"]] = video_dataset["Published_date"].astype(str).str.split("-", expand = True)
sorted_by_date=video_dataset.sort_values(by="Published_date")

sorted_by_date = sorted_by_date.astype({"year": int}, errors='raise') 
sorted_by_date = sorted_by_date.astype({"month": int}, errors='raise') 
sorted_by_date = sorted_by_date.astype({"day": int}, errors='raise') 

st.subheader('Insights on views')

Views=pd.DataFrame(video_dataset['Views'].describe())
st.write('Average views of this channel is', round(Views.at['mean','Views']))
st.write('Minimum views of this channel is', round(Views.at['min','Views']))
st.write('Maximum views of this channel is', round(Views.at['max','Views']))

st.subheader("Insights on likes")

Likes=pd.DataFrame(video_dataset['Likes'].describe())
st.write('Average likes of this channel is', round(Likes.at['mean','Likes']))
st.write('Minumum likes of this channel is', round(Likes.at['min','Likes']))
st.write('Maximum likes of this channel is', round(Likes.at['max','Likes']))

st.subheader("Insights on Comments")

Comments=pd.DataFrame(video_dataset['Comments'].describe())
st.write('Average comments of this channel is', round(Comments.at['mean','Comments']))
st.write('Minimum comments of this channel is', round(Comments.at['min','Comments']))
st.write('Maximum comments of this channel is', round(Comments.at['max','Comments']))

st.subheader('Graphical representation of the influence this channel has')
import matplotlib.pyplot as plt

def bar_chart():
    names = ['Views','Likes','Comments']
    values = [sum(video_dataset['Views']),sum(video_dataset['Likes']),sum(video_dataset['Comments'])]
    fig = plt.figure(figsize = (15, 5))
    plt.bar(names, values)
    plt.ylabel("Count")
    plt.title("Comparision of Views vs Likes vs Comments")
    st.pyplot(fig)
bar_chart()

names = ['Views','Likes']
values = [sum(video_dataset['Views']),sum(video_dataset['Likes'])]
fig1=plt.figure(figsize=(15,5))
plt.subplot(131)
plt.bar(names, values)
plt.ylabel("Count")
plt.suptitle('Total Views vs Likes')

names = ['Likes','Comments']
values = [sum(video_dataset['Likes']),sum(video_dataset['Comments'])]
fig2=plt.figure(figsize=(15,5))
plt.subplot(131)
plt.bar(names, values)
plt.ylabel("Count")
plt.suptitle('Total Likes vs Comments')

st.pyplot(fig1)
st.pyplot(fig2)


#put predictors here--------------------------------------------------------------------------------------------------------------------------------
st.subheader("Predictions")
df=video_dataset
date_diff=[]
for i in range(len(df)-1):
    if i <= len(df)-2:
        a=df['Published_date'][i]
        b=df['Published_date'][i+1]
        date_diff.append(abs((b-a).days))

next_date=df['Published_date'][0]+timedelta(round(sum(date_diff)/len(date_diff)))
diff=round(sum(date_diff)/len(date_diff))
if diff>1:
    st.write('This channel uploads new video after every {} days'.format(diff))
else:
    st.write('This channel uploads new video after every {} day'.format(diff))

data={'year':[next_date.year],
     'month':[next_date.month],
     'day':[next_date.day]}
A=pd.DataFrame(data)
model2=xgboost.XGBRegressor()
model2.load_model('project/View_model.json')
predicted_views=round(model2.predict(A)[0])
avg_views=round((sum(video_dataset.Views)/len(video_dataset.Views)))
if avg_views>predicted_views:
    st.write('This channel will get views anywhere between {} and {} on next video'.format(predicted_views,avg_views))
else:
    st.write('This channel will get views anywhere between {} and {} on next video'.format(avg_views,predicted_views))

comm = df_comments[:2000]
#Calculating the Sentiment Polarity--------------------------------------------------------------------------------------------------------------------------------
pol=[] # list which will contain the polarity of the comments
for i in comm.Comment.values:
    try:
        analysis =TextBlob(i)
        pol.append(analysis.sentiment.polarity)
        
    except:
        pol.append(0)

#Adding the Sentiment Polarity column to the data--------------------------------------------------------------------------------------------------------------------------------
comm['pol']=pol

#Converting the polarity values from continuous to categorical--------------------------------------------------------------------------------------------------------------------------------
comm['pol'][comm.pol==0]= 0
comm['pol'][comm.pol > 0]= 1
comm['pol'][comm.pol < 0]= -1
df_positive = comm[comm.pol==1]
df_negative = comm[comm.pol==-1]
df_neutral = comm[comm.pol==0]
# comm.pol.value_counts().pyplot.bar()--------------------------------------------------------------------------------------------------------------------------------
len_=len(comm)
positive_percentager=round((len(df_positive)/len(comm))*100)
negative_percentage=round((len(df_negative)/len(comm))*100)
neutral_percentage=round((len(df_neutral)/len(comm))*100)
st.subheader("Analysis of latest comments")
st.write('Percentage of the positive comments out of last {} comments is {} %'.format(len_,positive_percentager))
st.write('Percentage of the negative comments out of last {} comments is {} %'.format(len_,negative_percentage))
st.write('Percentage of the neutral comments out of last {} comments is {} %'.format(len_,neutral_percentage))


st.write('This channel has {}% positive influence on the audience in recent days.'.format(positive_percentager))


#check the content toxicity --------------------------------------------------------------------------------------------------------------------------------
# pip install youtube_transcript_api

from youtube_transcript_api import YouTubeTranscriptApi as yta
import re

vid_id=st.text_input('Please input a videoID to check the content quality of this channel')
st.write('Make sure the subtitles of the video are enabled')
data=yta.get_transcript(vid_id)

script_data=[]
for value in data:
    for key, val in value.items():
        if key=='text':
            script_data.append(val)

from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
model = tf.keras.models.load_model('project/toxicity.h5')
df = pd.read_csv('project/final_data.csv')
X = df['comment_text']
y = df[df.columns[3:]].values
MAX_FEATURES = 20
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1,
                               output_mode='int')

vectorizer.adapt(X.values)
input_str = vectorizer(script_data[0])
res = model.predict(np.expand_dims(input_str,0))

toxic=[]
severe_toxic=[]
obscene=[]
identity_hate=[]
insult=[]
threat=[]
funny=[]
wow=[]
sad=[]
sexual_explicit=[]
for i in script_data:
    input_str = vectorizer(i)
    res = model.predict(np.expand_dims(input_str,0))
    toxic.append(res[0][0])
    severe_toxic.append(res[0][1])
    obscene.append(res[0][2])
    identity_hate.append(res[0][3])
    insult.append(res[0][4])
    threat.append(res[0][5])
    funny.append(res[0][6])
    wow.append(res[0][7])
    sad.append(res[0][8])
    sexual_explicit.append(res[0][9])

st.subheader("Prediction of the Speech/Script quality of the video")
st.write('Toxicity: - {}%'.format(round((sum(toxic)/len(toxic))*5),3))
st.write('Severe Toxicity: - {}%'.format(round((sum(severe_toxic)/len(severe_toxic))*100),2))
st.write('Obscene: - {}%'.format(round((sum(obscene)/len(obscene))*100),2))
st.write('Identity hate: - {}%'.format(round((sum(identity_hate)/len(identity_hate))*100),2))
st.write('Insult: - {}%'.format(round((sum(insult)/len(insult))*100),2))
st.write('Threat: - {}%'.format(round((sum(threat)/len(threat))*100),2))
st.write('Funny: - {}%'.format(round((sum(funny)/len(funny))*100),2))
st.write('Wow: - {}%'.format(round((sum(wow)/len(wow))*100),2))
st.write('Sad: - {}%'.format(round((sum(sad)/len(sad))*100),2))
st.write('Sexual explicit: - {}%'.format(round((sum(sexual_explicit)/len(sexual_explicit))*100),2))

# --------------------------------------------------------------------------------------------------------------------------------

# Import the required libraries.
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from moviepy.editor import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from tensorflow import keras
LRCN_model=keras.models.load_model('project/LRCN_model___Date_Time_2022_07_11__17_20_47.h5')

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 128, 128

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 40

CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

def download_youtube_videos(youtube_video_url, output_directory):
     '''
    This function downloads the youtube video whose URL is passed to it as an argument.
    Args:
        youtube_video_url: URL of the video that is required to be downloaded.
        output_directory:  The directory path to which the video needs to be stored after downloading.
    Returns:
        title: The title of the downloaded youtube video.
    '''
 
     # Create a video object which contains useful information about the video.
     video = pafy.new(youtube_video_url)
 
     # Retrieve the title of the video.
     title = video.title
 
     # Get the best available quality object for the video.
     video_best = video.getbest()
 
     # Construct the output file path.
     output_file_path = f'{output_directory}/{title}.mp4'
 
     # Download the youtube video at the best available quality and store it to the contructed path.
     video_best.download(filepath = output_file_path, quiet = True)
 
     # Return the video title.
     return title
     

# Make the Output directory if it does not exist
test_videos_directory = 'project/test_videos'
os.makedirs(test_videos_directory, exist_ok = True)

def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    st.write(f'Action Predicted: {predicted_class_name}\nConfidence: {round((predicted_labels_probabilities[predicted_label]*100),2)}%')
        
    # Release the VideoCapture object. 
    video_reader.release()


# Download the youtube video.
video_title = download_youtube_videos('https://www.youtube.com/watch?v=wwbM63-LFD4&ab_channel=TechLead', test_videos_directory)

# Construct tihe nput youtube video path
input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'

# Perform Single Prediction on the Test Video.
predict_single_action(input_video_file_path, SEQUENCE_LENGTH)

# --------------------------------------------------------------------------------------------------------------------------------

# if st.checkbox('Delete this influencer'):
#     c.execute('DELETE FROM channelid_list WHERE chennel_id=?',([channelID]))
#     st.success('channel id {} is deleted successfully.'.format(channelID))

# # --------------------------------------------------------------------------------------------------------------------------------
