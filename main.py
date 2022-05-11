import pandas as pd
'''
data set columns: 'tweet_id', 'user_id', 'tweet_timestamp', 'keyword',
       'valence_intensity', 'fear_intensity', 'anger_intensity',
       'happiness_intensity', 'sadness_intensity', 'sentiment', 'emotion'
To access data in a column: data.colname (ex. data.tweet_id)
'''
data = pd.read_csv('small_dataset.csv')
data_no_neutral = data[data.emotion.ne('no specific emotion') & data.sentiment.ne('neutral or mixed')] #the and is redundant
data_no_wuhan = data_no_neutral[data_no_neutral.keyword.ne('corona') & data_no_neutral.keyword.ne('wuhan')]
#data_no_neutral = data[]
print(data_no_wuhan.keyword)
#print(data_no_neutral.emotion)