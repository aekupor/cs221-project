import itertools
import time
start_time = time.time()
import numpy
import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt #incase you guys want to do graphs in python
from bs4 import BeautifulSoup
pd.options.mode.chained_assignment = None  # default='warn'
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# SCRAPE WEBPAGE FOR EVENT TIMES
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
tp = pd.read_csv("tweetid_userid_keyword_sentiments_emotions_United States.csv", chunksize=50000,\
                 dtype = {'valence_intensity': 'float16', 'fear_intensity': 'float16', \
                          'anger_intensity': 'float16', 'happiness_intensity': 'float16', 'sadness_intensity': 'float16'},\
                 usecols=req_cols) #remove tweet and user id
start_idx = -1
end_idx = 0

# READ IN DATA WITH START TIMES
for chunk in tqdm(tp):
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

# GRAPHS OF DATA
means = df.groupby(['start times'])['fear_intensity'].mean() #fear over time
means.plot(x='start times', y='fear_intensity', kind = 'line', color= 'red', title="fear_intensity")
plt.show()

means = df.groupby(['start times'])['valence_intensity'].mean() #valence over time
means.plot(x='start times', y='valence_intensity', kind = 'line', color= 'red', title="valence_intensity")
plt.show()

means = df.groupby(['start times'])['anger_intensity'].mean() #anger over time
means.plot(x='start times', y='anger_intensity', kind = 'line', color= 'red', title="anger_intensity")
plt.show()

means = df.groupby(['start times'])['happiness_intensity'].mean() #happiness over time
means.plot(x='start times', y='happiness_intensity', kind = 'line', color= 'red', title="happiness_intensity")
plt.show()

means = df.groupby(['start times'])['sadness_intensity'].mean() #sadness over time
means.plot(x='start times', y='sadness_intensity', kind = 'line', color= 'red', title="sadness_intensity")
plt.show()

# LINEAR REGRESSION

# convert dates into floats of days past first date
df['date'] = pd.to_datetime(df['start times'])
df['date_delta'] = (df['date'] - df['date'].min()) / np.timedelta64(1,'D')

# separate the other attributes that we don't want
x = df.drop(['start times', 'date', 'tweet_timestamp', 'keyword', 'emotion', 'sentiment', 'date_delta'], axis=1)
# print(x) # uncomment this line to visualize data
#separte the predicting attribute into Y for model training
y = df['date_delta']

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
LR = LinearRegression()

# fitting the training data
LR.fit(x_train,y_train)
y_prediction =  LR.predict(x_test)

# print prediction and coeffiencts
print(y_prediction)
print(LR.coef_)

#print our prediction vs actual
print("MATMUL")
print(np.matmul(x_test, LR.coef_))
print("YTEST")
print(y_test)