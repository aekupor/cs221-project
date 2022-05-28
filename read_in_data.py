import time
start_time = time.time()
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm

# SCRAPE WEBPAGE FOR EVENT TIMES
with open("Covid-19 Pandemic Timeline Fast Facts - CNN.html") as fp:
    soup = BeautifulSoup(fp, "html.parser")
times = soup.find_all("strong")
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
convertedTimes = []
tempTimes = []
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
        temp_date = monthNum+dayNum
        pd.to_datetime(dateStamp)
        tempTimes.append(temp_date)
        convertedTimes.append(dateStamp)
tempTimes = list(set(tempTimes))
tempTimes = sorted(tempTimes)
convertedTimes = list(set(convertedTimes)) #remove duplicates
convertedTimes = sorted(convertedTimes)
convertedTimes = np.array(convertedTimes)
convertedTimes = convertedTimes.astype("datetime64")

notable = pd.read_csv("Notable_Dates.csv")
date_notable = notable['Date'].to_numpy()
date_notable = date_notable.astype("datetime64")
#print(date_notable)
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
new_req_cols = ['tweet_timestamp',
       'valence_intensity', 'fear_intensity', 'anger_intensity',
       'happiness_intensity', 'sadness_intensity']
# req_cols = ['tweet_timestamp', 'keyword',
#        'valence_intensity', 'fear_intensity', 'anger_intensity',
#        'happiness_intensity', 'sadness_intensity', 'sentiment', 'emotion']
tp = pd.read_csv("tweetid_userid_keyword_sentiments_emotions_United States.csv", chunksize=50000,\
                 dtype = {'valence_intensity': 'float16', 'fear_intensity': 'float16', \
                          'anger_intensity': 'float16', 'happiness_intensity': 'float16', 'sadness_intensity': 'float16'},\
                 usecols=new_req_cols) #remove tweet and user id
start_idx = -1
end_idx = 0
start_idx_not = -1
end_idx_not = 0
# labeling data with index of event/event name
# READ IN DATA WITH START TIMES
for chunk in tqdm(tp):
    #not needed with new_req_cols chunk_filtered = chunk[chunk.emotion.ne('no specific emotion')] #remove neutral data
    chunk['tweet_timestamp'] = chunk['tweet_timestamp'].str.split().str[0] #remove time and just leave date
    date_arr = chunk['tweet_timestamp'].to_numpy()
    date_arr = date_arr.astype("datetime64")
    valid_dates = np.array([], dtype="datetime64") #tweets with time stamp that falls in the range
    start_times = []
    start_times_notable = []
    idx_not_arr = []
    #work to label data with normal/general start time
    while len(start_times) < date_arr.size:
        valid_dates = np.array([], dtype="datetime64")  # tweets with time stamp that falls in the range
        mask = (date_arr >= convertedTimes[start_idx]) & (date_arr < convertedTimes[end_idx])
        if date_arr[mask].size > 0:
            valid_dates = np.concatenate((valid_dates, date_arr[mask]))
        bigger_mask = (date_arr >= convertedTimes[end_idx])
        if date_arr[bigger_mask].size > 0:  # if one of the dates exceeds the end time of the range then we need to increment the indices
            start_idx = end_idx
            end_idx = end_idx + 1
        if valid_dates.size > 0:
            start_time = convertedTimes[start_idx]
            for i in range(valid_dates.size):
                start_times.append(start_time)
    #work to label data with notable start time
    while len(start_times_notable) < date_arr.size:
        valid_dates_notable = np.array([], dtype="datetime64")
        early_dates_notable = np.array([], dtype="datetime64")
        mask_notable = (date_arr >= date_notable[start_idx_not]) & (date_arr < date_notable[end_idx_not])
        start_times_notable = [] #NO IDEA WHY THIS IS NECESSARY
        idx_not_arr = []
        if date_arr[mask_notable].size > 0:
            valid_dates_notable = np.concatenate((valid_dates_notable, date_arr[mask_notable]))
        bigger_mask_not = (date_arr >= date_notable[end_idx_not])
        smaller_mask_not = (date_arr < date_notable[start_idx_not])
        if date_arr[bigger_mask_not].size > 0: #if one of the dates exceeds the end time of the range then we need to increment the indices
            start_idx_not = end_idx_not
            end_idx_not = end_idx_not + 1
        if date_arr[smaller_mask_not].size > 0: #if earliest tweet time stamp preceeds earliest notable date
            early_dates_notable = np.concatenate((early_dates_notable, date_arr[smaller_mask_not]))
        if early_dates_notable.size > 0:
            early = None
            for i in range(early_dates_notable.size):
                start_times_notable.append(early)
                idx_not_arr.append(early)
        if valid_dates_notable.size > 0:
            start_time_not = date_notable[start_idx_not]
            for i in range(valid_dates_notable.size):
                start_times_notable.append(start_time_not)
                idx_not_arr.append(start_idx_not)
    chunk['start times'] = start_times
    chunk['start notable'] = start_times_notable
    chunk[' idx notable'] = idx_not_arr
    #convert notable start times to a numpy date time array
    start_times_notable = np.array(start_times_notable)
    start_times_notable = start_times_notable.astype('datetime64')
    chunk['days since'] = date_arr - start_times_notable #calculate difference in days
    chunk_list.append(chunk)
df = pd.concat(chunk_list, ignore_index=True)

df.to_pickle('data_pickle.pkl')
