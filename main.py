import itertools

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #incase you guys want to do graphs in python
from bs4 import BeautifulSoup
pd.options.mode.chained_assignment = None  # default='warn'

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
            monthNum = f'0{monthNum}'
        else:
            monthNum = f'{monthNum}'
        dayNum = time[1][:-1]
        if int(dayNum) < 10:
            dayNum = '0' + dayNum
        else:
            dayNum = dayNum
        yearNum = time[2]
        dateStamp = yearNum + '-' + monthNum + '-' + dayNum #+ ' ' + '00-00-00'
        pd.to_datetime(dateStamp)
        convertedTimes.append(dateStamp)
convertedTimes = list(set(convertedTimes)) #remove duplicates
convertedTimes = sorted(convertedTimes)

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
tp = pd.read_csv("small_dataset.csv", chunksize=10,\
                 dtype = {'valence_intensity': 'float16', 'fear_intensity': 'float16', \
                          'anger_intensity': 'float16', 'happiness_intensity': 'float16', 'sadness_intensity': 'float16'},\
                 usecols=req_cols) #remove tweet and user id
start_idx = -1
end_idx = 0
for chunk in tp:
    chunk_filtered = chunk[chunk.emotion.ne('no specific emotion')] #remove neutral data
    chunk_filtered['tweet_timestamp'] = chunk_filtered['tweet_timestamp'].str.split().str[0] #remove time and just leave date
    convertedTimes = np.array(convertedTimes)
    convertedTimes = convertedTimes.astype("datetime64")
    date_arr = chunk_filtered['tweet_timestamp'].to_numpy()
    date_arr = date_arr.astype("datetime64")
    valid_dates = np.array([], dtype="datetime64") #tweets with time stamp that falls in the range
    start_times = []
    while len(start_times) < date_arr.size:
        valid_dates = np.array([], dtype="datetime64")  # tweets with time stamp that falls in the range
        mask = (date_arr >= convertedTimes[start_idx]) & (date_arr < convertedTimes[end_idx])
        if date_arr[mask].size > 0:
            valid_dates = np.concatenate((valid_dates, date_arr[mask]))
        bigger_mask = (date_arr >= convertedTimes[end_idx])
        if date_arr[bigger_mask].size > 0: #if one of the dates exceeds the end time of the range then we need to increment the indices
            start_idx = end_idx
            end_idx = end_idx + 1
        if valid_dates.size > 0:
            start_time = convertedTimes[start_idx]
            for i in range(valid_dates.size):
                start_times.append(start_time)
    chunk_filtered['start times'] = start_times
    chunk_list.append(chunk_filtered)
df = pd.concat(chunk_list, ignore_index=True)
#print(df)



# date_arr = numpy.datetime64(date_arr)
    # pd.to_datetime(chunk_filtered['tweet_timestamp'])
    # curr_idx = 0 #index of current time start time
    # next_idx = 1 #index of end time in converted times
    # starttimes = []
    # for index, row in chunk_filtered.iterrows():
    #     timeStamp = row["tweet_timestamp"]
    #    # startTime = convertedTimes[curr_idx]
    #     #endTime = convertedTimes[next_idx]
    #     while timeStamp <= convertedTimes[curr_idx]:
    #         print('hi')
    #         if convertedTimes[next_idx] < timeStamp:
    #             curr_idx = next_idx
    #             next_idx = next_idx + 1
    #         elif convertedTimes[curr_idx] <= timeStamp < convertedTimes[next_idx]:
    #             starttimes.append(convertedTimes[curr_idx])
    # filtered_times = chunk_filtered['tweet_timestamp'].to_numpy()
    # np.datetime64(filtered_times)
    # print(filtered_times)
    #for row in chunk_filtered:


    # for index, date in enumerate(convertedTimes):
    #     start
    #     if
# for chunk in tp:
#     chunk_filtered = chunk[chunk.emotion.ne('no specific emotion')] #remove neutral data
#     pd.to_datetime(chunk_filtered['tweet_timestamp'])
#     chunk_filtered = chunk_filtered.reset_index()
#     start_times = []
#     for index,row in chunk_filtered.iterrows():
#         for date in convertedTimes:
#             start_date = date
#             endIdx = convertedTimes.index(date) + 1
#             if endIdx < len(convertedTimes): #correct?
#                 end_date = convertedTimes[endIdx]
#                 if start_date <= row['tweet_timestamp'] < end_date:
#                     start_times.append(start_date)
#                     del convertedTimes[0:(endIdx - 1)]
#                     break


        # old code with out deleting unnecessary first part of converted times
        # for index in range(len(convertedTimes)-1):
        #     start_date = convertedTimes[index]
        #     end_date = convertedTimes[index + 1]
        #     if start_date <= row['tweet_timestamp'] < end_date:
        #         start_times.append(start_date)
#     chunk_filtered['start times'] = starttimes
#     chunk_list.append(chunk_filtered)
# df = pd.concat(chunk_list, ignore_index=True)
#
