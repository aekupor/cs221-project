import pandas as pd
import matplotlib.pyplot as plt #incase you guys want to do graphs in python
from bs4 import BeautifulSoup

#Scrape webpage for event times
with open("Covid-19 Pandemic Timeline Fast Facts - CNN.html") as fp:
    soup = BeautifulSoup(fp, "html.parser")
times = soup.find_all("strong")
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
convertedTimes = []
for time in times:
    time = time.text
    time = time.split()
    if len(time) > 1:
        monthNum = months.index(time[0]) + 1
        if monthNum < 10:
            print(type(monthNum))
            monthNum = f'0{monthNum}'
        else:
            monthNum = f'{monthNum}'
        dayNum = time[1][:-1]
        if int(dayNum) < 10:
            dayNum = '0' + dayNum
        else:
            dayNum = dayNum
        yearNum = time[2]
        dateStamp = yearNum + '-' + monthNum + '-' + dayNum + ' ' + '00-00-00'
        convertedTimes.append(dateStamp)
print(convertedTimes)

#sources
#https://towardsdatascience.com/%EF%B8%8F-load-the-same-csv-file-10x-times-faster-and-with-10x-less-memory-%EF%B8%8F-e93b485086c7
#https://stackabuse.com/guide-to-parsing-html-with-beautifulsoup-in-python/
#read in data sequentially and preprocess (remove neutral, make int type smaller, remove tweet and user id)
'''
data set columns: 'tweet_id', 'user_id', 'tweet_timestamp', 'keyword',
       'valence_intensity', 'fear_intensity', 'anger_intensity',
       'happiness_intensity', 'sadness_intensity', 'sentiment', 'emotion'
To access data in a column: data.colname (ex. data.tweet_id)
'''
chunk_list = []
req_cols = ['tweet_timestamp', 'keyword',
       'valence_intensity', 'fear_intensity', 'anger_intensity',
       'happiness_intensity', 'sadness_intensity', 'sentiment', 'emotion']
tp = pd.read_csv("tweetid_userid_keyword_sentiments_emotions_United States.csv", chunksize=50000,\
                 dtype = {'valence_intensity': 'float16', 'fear_intensity': 'float16', \
                          'anger_intensity': 'float16', 'happiness_intensity': 'float16', 'sadness_intensity': 'float16'},\
                 usecols=req_cols) #remove tweet and user id
for chunk in tp:
    chunk_filtered = chunk[chunk.emotion.ne('no specific emotion')] #remove neutral data
    chunk_list.append(chunk_filtered)
    print(chunk_filtered)
df = pd.concat(chunk_list, ignore_index=True)

