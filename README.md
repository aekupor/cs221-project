# Covid Tweet Sentiment Classification
Extra credit project for CS221 class.

### Commands to run
1. Download any packages you do not already have (scikit, numpy, pandas, matplotlib, etc.)
2. Download the US tweet dataset to your local computer from this [link](https://www.openicpsr.org/openicpsr/project/120321/version/V11/view;jsessionid=F844CFFA986C0B1E2D913D34A99C9CDE?path=/openicpsr/120321/fcr:versions/V11/Twitter-COVID-dataset---Sep2021&type=folder)
3. Get the dataset you want (either 1st or 2nd) by running 

```
python3 read_in_data.py 
```
or 
```
python3 read_in_data_2.py
```
, respecitvely.

<br /> 4. Comment out what you do not run to run in main.py (graphs, linear regression, which dataset to read in) and then run:

```
python3 main.py 
```

### Overview of Code
- read_in_data.py and read_in_data_2.py contain code to read in the data. This file only needs to be run once on your local computer.
- main.py contains most of the code:
    - print_graphs() prints out the graphs for the data
    - linear_regression() runs the linear regression on a modified datafram
    - main() reads in the data from the pickle, edits the dataframe (by grouping tweets or filtering to a number of days post time periods), and then calls the graph or linear regression functions
