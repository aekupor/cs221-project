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
print(date_notable)
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
        if valid_dates_notable.size > 0:
            start_time_not = date_notable[start_idx_not]
            for i in range(valid_dates_notable.size):
                start_times_notable.append(start_time_not)
    chunk['start times'] = start_times
    chunk['start notable'] = start_times_notable
    #convert notable start times to a numpy date time array
    start_times_notable = np.array(start_times_notable)
    start_times_notable = start_times_notable.astype('datetime64')
    chunk['days since'] = date_arr - start_times_notable #calculate difference in days
    chunk_list.append(chunk)
df = pd.concat(chunk_list, ignore_index=True)
print(df)

# GRAPHS OF DATA
ax = plt.axes()
means = df.groupby(['start times'])['fear_intensity'].mean() #fear over time
means.plot(x='start times', y='fear_intensity', kind = 'line', color= 'red', title="fear_intensity")
# setting ticks for x-axis
#ax.set_xticks(tempTimes)
plt.show()

ax = plt.axes()
means = df.groupby(['start times'])['valence_intensity'].mean() #valence over time
means.plot(x='start times', y='valence_intensity', kind = 'line', color= 'red', title="valence_intensity")
#ax.set_xticks(tempTimes)
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