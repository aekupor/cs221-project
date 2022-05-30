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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def print_graphs():
    pass
    #
    # # GRAPHS OF DATA
    # ax = plt.axes()
    # means = df.groupby(['start times'])['fear_intensity'].mean() #fear over time
    # means.plot(x='start times', y='fear_intensity', kind = 'line', color= 'red', title="fear_intensity")
    # # setting ticks for x-axis
    # #ax.set_xticks(tempTimes)
    # plt.show()
    #
    # ax = plt.axes()
    # means = df.groupby(['start times'])['valence_intensity'].mean() #valence over time
    # means.plot(x='start times', y='valence_intensity', kind = 'line', color= 'red', title="valence_intensity")
    # #ax.set_xticks(tempTimes)
    # plt.show()
    #
    # means = df.groupby(['start times'])['anger_intensity'].mean() #anger over time
    # means.plot(x='start times', y='anger_intensity', kind = 'line', color= 'red', title="anger_intensity")
    # plt.show()
    #
    # means = df.groupby(['start times'])['happiness_intensity'].mean() #happiness over time
    # means.plot(x='start times', y='happiness_intensity', kind = 'line', color= 'red', title="happiness_intensity")
    # plt.show()
    #
    # means = df.groupby(['start times'])['sadness_intensity'].mean() #sadness over time
    # means.plot(x='start times', y='sadness_intensity', kind = 'line', color= 'red', title="sadness_intensity")
    # plt.show()
    #

def linear_regression(df, needToDrop = True):
    # separate the other attributes that we don't want
    if needToDrop:
        x = df.drop(['start times', 'tweet_timestamp', 'days since', 'start notable', ' idx notable'], axis=1)
    else:
        x = df.drop([' idx notable'], axis=1)
    #print(x) # uncomment this line to visualize data
    # separte the predicting attribute into Y for model training
    y = df[' idx notable']

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    LR = LinearRegression()

    # fitting the training data
    LR.fit(x_train,y_train)
    y_prediction =  LR.predict(x_test)

    # print prediction and coeffiencts
    # print(y_prediction)
    # print(LR.coef_)

    #print our prediction vs actual
    # print("MATMUL")
    # print(np.matmul(x_test, LR.coef_))
    # print("YTEST")
    # print(y_test)

    print("ACCURACY")
    y_pred_rounded = [round(float(x)) for x in y_prediction]
    y_test_rounded = [round(float(x)) for x in y_test]
    print(accuracy_score(y_test_rounded, y_pred_rounded))
    # print(y_test_rounded)
    print("PRECISION")
    print(precision_score(y_test_rounded, y_pred_rounded,  average = 'micro'))
    print("FI Score")
    print(f1_score(y_test_rounded, y_pred_rounded, average = 'micro'))


def main():
    # READ IN DATA
    # df = pd.read_pickle('data_pickle.pkl') # first set of dates
    df = pd.read_pickle('data_pickle_2.pkl')

    # LINEAR REGRESSION

    df = df.dropna()
    # Riya Note: use 3 for first dataset in number of days
    # EMILY CHANGE HERE: 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5
    df = df[df['days since'] <= pd.Timedelta(2,'D')]
    df_grouped = df

    print("REGULAR")
    # linear_regression(df) # uncomment if you want to run without grouping
    # EMILY CHANGE HERE: 5, 10, 50, 100, 10000, 5000, 100000
    df_grouped = df_grouped.groupby(np.arange(len(df_grouped))//10000).mean()
    # print(df_grouped)
    print("GROUPED")
    linear_regression(df_grouped, needToDrop = False)

    # GRAPHS
    print_graphs()

if __name__ == "__main__":
    main()