# Alicia Domingo
# 12/13/22
# Final Project

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree

# FUNCTIONS

# Decription: Converting the duration column from seconds to hours, with the input of the dataframe
def seconds_to_hours(df):
    duration_col = df["DURATION"]
    
    for i in range(len(df)):
        duration_col[i] = duration_col[i]/3600
        
    df["DURATION"] = duration_col
        
# Desciption: Coverting the date column to a standard date, removing the time, with the input of dataframe
def convert_date_to_std (df):
    df["START DATE(UTC)"] = pd.to_datetime(df['START DATE(UTC)']).dt.date
    df["END DATE(UTC)"] = pd.to_datetime(df['END DATE(UTC)']).dt.date
   
# Desciption: Being able to visualize all the data from life cycle with a pie chart with the input of dataframe 
def overall_pie(df):
    grouped_by_name = df.groupby("NAME")
    duration_ser = grouped_by_name["DURATION"].sum()
    duration_ser = round(duration_ser,2)
    sorted_ascending = duration_ser.sort_values(ascending=True)
    sorted_ascending = sorted_ascending.to_frame()

    duration = sorted_ascending["DURATION"]
    duration_250_plus = duration.iloc[56:71]
    
    plt.pie(duration_250_plus,labels=duration_250_plus.index, radius=2.5,startangle=90)
    
# Desciption: Creating a bar chart of the time spent on home depending on the year, with the dataframe data, but also the 
#               start and finish of each year from the csv file.
def time_spent_home_bar (df,start,finish,year):
    year_2018 = df.iloc[start:finish]
    grouped_by_name = year_2018.groupby("NAME")
    home_df = grouped_by_name.get_group(" Home")

    duration = home_df["DURATION"]
    mean = home_df["DURATION"].mean()
    date = home_df["START DATE(UTC)"]
    print("Average:",mean)

    plt.bar(date,duration,color='blue',width=2.5)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Time from " + str(year))
    plt.ylabel("Hours")
    plt.title("Time spent at Home in " + str(year))
    
# Decription: Getting the necessary data for the time spent sleeping by using the split, combine, and apply method
def time_spent_sleep (df, start, finish, month):
    month_2022 = df.iloc[start:finish]
    grouped_by_name = month_2022.groupby("NAME")
    sleep_month_df = grouped_by_name.get_group(" Sleep")
    
    duration = sleep_month_df["DURATION"]
    
    return duration, sleep_month_df

# Creating a visualization of the time I spend sleeping using a bar chart, with also the mean and standard deviation
def visualize_sleep(df,start,finish,month):
    sleep_funct = time_spent_sleep(df,start,finish,month)
    sleep_month_df = sleep_funct[1]
    
    duration = sleep_month_df["DURATION"]
    date = sleep_month_df["START DATE(UTC)"]
    
    avg_sleep_month = sleep_month_df["DURATION"].mean()
    std_sleep_month = sleep_month_df["DURATION"].std()
    
    print("Average Sleep in " + month, avg_sleep_month)
    print("Standard Deviation in " + month, std_sleep_month)
    
    plt.bar(date,duration, color='purple')
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Time from " + month)
    plt.ylabel("Hours")
    plt.title("Time spent sleeping in " + month)

# Desciption: Using hypothesis testing of the time I spent sleeping in June and September of 2022, to see it I got more sleep
#               during the summer
def hypothesis_testing(df,alpha):
    sleep_june_duration = time_spent_sleep(df,16119,16272,"June")
    sleep_september_duration = time_spent_sleep(df,16759,17172,"September")
    
    sleep_june_duration = sleep_june_duration[0]
    sleep_september_duration = sleep_september_duration[0]
    
    length_june = len(sleep_june_duration)
    length_september = len(sleep_september_duration)
    degrees_freedom = length_june + length_september - 2
    t_critical = 2.666

    print("Degrees of Freedom:",degrees_freedom)
    print("T-critical:", t_critical)
    
    t, pval = stats.ttest_rel(sleep_june_duration,sleep_september_duration)

    if pval <= alpha:
        print("reject H0")
    else:
        print("do not reject H0")

# Desciption: converting date to a string of just numbers, removing the "-"
def replace_date (df):
    df['START DATE(UTC)'] = df['START DATE(UTC)'].astype(str)
    df["START DATE(UTC)"] = df["START DATE(UTC)"].str.replace("-","")
    df['START DATE(UTC)'] = df['START DATE(UTC)'].astype(int)

# Decription: Changing the name column into just numbers, to be prepared for classification
def categorical_to_numeric (df):
    le = preprocessing.LabelEncoder()
    le.fit(df["NAME"])
    list(le.classes_)
    df["NAME"] = le.transform(df["NAME"])

    df.to_csv("project_knn.csv")

# Decription: Dividing the dataset into training and testing for classification
def for_train_test_split(df):
    X = df.drop("NAME",axis=1)
    y = df["NAME"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)
    
    return X_train, X_test, y_train, y_test

# Description: Getting the accuracy using the KNN classifier
def accuracy(df):
    tts = for_train_test_split(df)
    
    X_train = tts[0]
    X_test = tts[1]
    y_train = tts[2]
    y_test = tts[3]
    
    knn_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_clf.fit(X_train, y_train)
    acc = knn_clf.score(X_test, y_test)
    print("accuracy:", acc)

    y_predicted = knn_clf.predict(X_test)
    print(y_predicted)
    acc = accuracy_score(y_test, y_predicted)
    print("accuracy:", acc)
    
# Description: Getting the accuracy for the Decision tree classifier
def tree_class(df):
    tts = for_train_test_split(df)
    
    X_train = tts[0]
    X_test = tts[1]
    y_train = tts[2]
    y_test = tts[3]
    
    tree_clf = DecisionTreeClassifier(random_state=0)
    tree_clf.fit(X_train, y_train)
    acc = tree_clf.score(X_test, y_test)
    print("accuracy:", acc)
    
    clf = tree.DecisionTreeClassifier(random_state=0)
    iris = load_iris()
    clf = clf.fit(iris.data, iris.target)
    tree.plot_tree(clf, filled=True)

# Description: Merging two files together to make a new dataframe
def merge_files(df,df1):
    grouped_by_name = df.groupby("NAME")
    home_df = grouped_by_name.get_group(" Home")

    grouped_by_date = home_df.groupby("START DATE(UTC)")
    duration_ser = grouped_by_date["DURATION"].sum()
    duration_df = duration_ser.to_frame()
    duration_df.reset_index(inplace=True)

    merged_df = pd.merge(duration_df, df1, left_index=True, right_index=True)
    merged_df = merged_df.drop("Start Date(UTC)", axis=1)
    merged_df.to_csv("merged.csv")

    grouped_by_date = merged_df.groupby("DAY OF THE WEEK")
    duration_ser = grouped_by_date["DURATION"].sum()
    duration_df = duration_ser.to_frame()
    
    return duration_df

# Description: To visualize the merged dataframe
def visualize_merge(df):
    plt.bar(df.index,df["DURATION"],color="green",edgecolor='black')
    plt.xticks(rotation=45, ha="right")
    plt.title("Hours Spent at Home each Day")
    plt.xlabel("Days of the Week")
    plt.ylabel("Time(Hours)")