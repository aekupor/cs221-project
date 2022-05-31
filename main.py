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

def print_graphs(df):
    #
    # # GRAPHS OF DATA
    # ax = plt.axes()
    df.groupby(['start times'])['fear_intensity'].mean().plot(legend = True) #fear over time
    # means.plot(x='start times', y='fear_intensity', kind = 'line', color= 'red', title="fear_intensity")
    # # setting ticks for x-axis
    # #ax.set_xticks(tempTimes)
    # plt.show()
    #
    # ax = plt.axes()
    df.groupby(['start times'])['valence_intensity'].mean().plot(legend=True) #valence over time
    # means.plot(x='start times', y='valence_intensity', kind = 'line', color= 'red', title="valence_intensity")
    # #ax.set_xticks(tempTimes)
    # plt.show()
    #
    df.groupby(['start times'])['anger_intensity'].mean().plot(legend=True) #anger over time
    # means.plot(x='start times', y='anger_intensity', kind = 'line', color= 'red', title="anger_intensity")
    # plt.show()
    #
    df.groupby(['start times'])['happiness_intensity'].mean().plot(legend=True) #happiness over time
    # means.plot(x='start times', y='happiness_intensity', kind = 'line', color= 'red', title="happiness_intensity")
    # plt.show()
    #
    df.groupby(['start times'])['sadness_intensity'].mean().plot(legend=True) #sadness over time
    # means.plot(x='start times', y='sadness_intensity', kind = 'line', color= 'red', title="sadness_intensity")
    plt.annotate("WHO names COVID-19", xy = ('2020-02-11', 0.40), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate("Italy lockdown", xy = ('2020-02-23', 0.39), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate("WHO declared Covid-19 a pandemic", xy = ('2020-03-11', 0.45), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate("Senate passes CARES act", xy = ('2020-03-26', 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate("White house extendes social distancing", xy = ('2020-03-28', 0.37), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate("CDC recommends masking at all times", xy = ('2020-4-03', 0.42), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate("President Trump tests positive for the coronavirus.", xy = ('2020-9-22', 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate("U.S.COVID-19 death toll surpasses 200000.", xy = ('2020-9-22', 0.4), arrowprops=dict(facecolor='black', shrink=0.05))


    plt.annotate("Delta Variant Becomes Dominant.", xy = ('2021-8-03', 0.4), xytext = ('2021-6-03', 0.4), arrowprops=dict(facecolor='red', shrink=0.05))
    plt.annotate("Vaccine Rollout begins.", xy = ('2020-12-14', 0.33), arrowprops=dict(facecolor='red', shrink=0.05))

    plt.legend(loc = 4)
    # plt.text("2020-03-11", 0.5, "WHO declares COVID-19 as a pandemic")
    # plt.text("2020-03-13", 0.5, "Trump declares nationwide emergency")
    # plt.text("2020-03-26", 0.3, "Senate passes CARES Act")
    # plt.text("2020-03-28", 0.5,	"White House extends social distancing measures.")
    # plt.text("2020-04-03", 0.3,	"CDC recommends masking all the time.")
    # plt.text("2020-09-22", 0.5,	"United States coronavirus (COVID-19) death toll surpasses 200000.")
    # plt.text("2020-10-02", 0.3, "President Trump tests positive for the coronavirus.")
    # plt.text("2021-02-21", 0.5,	"U.S.COVID-19 death toll surpasses 500000.")
    # plt.text("2021-02-27", 0.3,	"FDA approves emergency use authorization for Johnson and Johnson one shot COVID-19 vaccine.")
    plt.show()
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

    return accuracy_score(y_test_rounded, y_pred_rounded)


def main():
    # READ IN DATA
    df = pd.read_pickle('data_pickle.pkl') # first set of dates
    #df = pd.read_pickle('data_pickle_2.pkl')

    # LINEAR REGRESSION

    df = df.dropna()
    # Riya Note: use 3 for first dataset in number of days
    x_values = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    # EMILY CHANGE HERE: 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5
    #df0 = df[df['days since'] <= pd.Timedelta(2,'D')]
    dfs = []
    for x in x_values:
        df = pd.read_pickle('data_pickle_2.pkl') 
        df = df.dropna()
        df = df[df['days since'] <= pd.Timedelta(x,'D')]
        dfs.append(df)
    #df_grouped = df

    print("REGULAR")
    y_values =[]
    for df in dfs:
        df_grouped = df
        df_grouped = df_grouped.groupby(np.arange(len(df_grouped))//20000).mean()
        y_values.append(linear_regression(df_grouped, needToDrop = False))
    plt.plot(x_values, y_values)
    plt.xlabel("Days since event")
    plt.ylabel("Accuracy")
    plt.show()


    
    # x_values = [5, 10, 50, 100, 1000, 5000, 10000, 20000, 30000, 50000]
    # y_values = []
    # dfs_grouped = []
    # for x in x_values:
    #     df = pd.read_pickle('data_pickle_2.pkl') 
    #     df = df.dropna()
    #     df = df[df['days since'] <= pd.Timedelta(2,'D')]
    #     df_grouped = df
    #     df_grouped = df_grouped.groupby(np.arange(len(df_grouped))//x).mean()
    #     dfs_grouped.append(df_grouped)
    # # print(df_grouped)
    # print("GROUPED")
    # for df_grouped in dfs_grouped:
    #     y_values.append(linear_regression(df_grouped, needToDrop = False))
    # plt.plot(x_values, y_values)
    # plt.xlabel("Grouping")
    # plt.ylabel("Accuracy")
    # plt.show()


    # GRAPHS
    #print_graphs(df)

if __name__ == "__main__":
    main()