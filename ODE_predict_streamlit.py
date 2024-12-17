import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from ODE_predict_helpers import *
import warnings
warnings.filterwarnings('ignore')


##################### Prepare data ############################

#Import data and choose predictors
grade_data = pd.read_excel('/Users/gyoung19/Library/CloudStorage/OneDrive-KennesawStateUniversity/Python/ODE_Predict/Grade_data/ODE_Grade_Data.xlsx')

#Keep only early grade data
early_data = grade_data[['Semester', 'HW1', 'HW2', 'HW3', 'HW4', 'Quiz1', 'Quiz2', 'Quiz3', 'Midterm1', 'Course_Grade']]
#Insert quiz and homework averages (normalized between 0 and 1)
early_data.insert(5, 'HW Avg',early_data[['HW1','HW2','HW3','HW4']].mean(axis=1)/10)
early_data.insert(9, 'Quiz Avg',early_data[['Quiz1','Quiz2','Quiz3']].mean(axis=1)/10)

#y is the target vector: 0 if the student failed, 1 if they passed
y = 1 - ((grade_data['Course_Grade'] == 'D') + (grade_data['Course_Grade'] == 'F'))*1

early_data.insert(loc = len(early_data.columns), column = 'Pass', value = y)

X = early_data[['HW Avg', 'Midterm1', 'Quiz Avg']]
X.insert(loc = len(X.columns), column = 'Weighted Avg', value = 0.1*X['HW Avg'] + 0.25*X['Quiz Avg'] + 0.65*X['Midterm1'])

# Delete grade_data if space is a problem (shouldn't matter since grade_data isn't too big)
# del grade_data

yes_str = 'Yes'
#################### Train Models ##########################


some_features = ['Midterm1', 'Quiz Avg']
lr_some = model_train(X, y, some_features, balance=None)

avg_features = ['Weighted Avg']
lr_avg = model_train(X, y, avg_features, balance=None)


#################### Set up Streamlit page ###########################
st.set_page_config(layout="wide")

st.write('# Math 2306 Course Outcome Predictor')


st.write('''### **Important information**:
- This tool will provide you with a (estimated) probability of passing Math 2306 based on your early-semester performance.
- The probability is generated using a logistic regression model trained on data from past Math 2306 students.
- The model was trained only on grade data from students who completed the course. Students who withdrew before the end of the semester were omitted from the training dataset.
- The point of this tool is to give you a sense of how students with similar early-semester performance have done in the course to help you make a decision about withdrawing from the course.
- No matter the generated probability, *you are not guaranteed to pass or fail this course.* Your success in the course is dependent on how much effort you put in.
- This is only intended to be used by students in Dr. Glenn Young's Math 2306 course at Kennesaw State University. The model will not give meaningful results to students in other classes.
''')

st.write('''## **How to use**:
- Choose which model you'd like to use. Both models give you an estimated probability of passing the course. One model makes the prediction based on your early-semester course average, the other makes the prediction based on your quiz average and midterm 1 score.
- Enter the requested grade information. 
    - The weighted course average should include your average score on the first four homework sets, the first three quizzes, and the first midterm.
    - The quiz average is the average of your first three quizzes.
- Read the estimated probability of passing the course and the associated table and figure.
- It's probably best to interpret the generated probability in terms of frequency. For example, if the model predicts you have a 75% probability of passing, then the model expects 3 out of 4 students in your position will pass.
''')

opt1 = "Weighted course average"
opt2 = "Quiz average and Midterm 1 score"

st.write('## Input early-semester grade data.')
st.write('#### First, choose whether you\'d like to use your early-semester course average or a combination of your midterm 1 and quiz average.')
option = st.selectbox(
    "**What grades would you like to use to predict course outcome?**",
    (opt1, opt2),
    index=0
)


#st.write('Your selection:', option)
if option == opt1:
    st.write('**Please enter weighted course average using the following formula: 10\*HW + 25\*Quiz + 65\*Midterm1.**')
    new_avg = st.number_input('**Please input your average as a decimal (e.g., 0.71, not 71).**', min_value=0.00, max_value=1.00, value='min')
    #new_avg=0.6
    df_closest = find_closest_1D(new_avg, 10, X['Weighted Avg'], y)
    df_closest.index = np.arange(1, len(df_closest)+1)

    st.write('## Model output.')
    con = st.container(border=True)
    con.write(f'#### The logistic regression model estimates that you have a {lr_avg.predict_proba([[new_avg]])[0,1]*100:.03}% probability of passing the course.')

    left_column, right_column = st.columns(2)
    # You can use a column just like st.sidebar:
    left_column.write('Here are the ten closest grades and their course outcomes:')
    left_column.dataframe(df_closest, use_container_width=True)
    left_column.write(f'Among the students with the 10 closest averages to yours, {np.sum([x==yes_str for x in df_closest.Passed])} passed the course.')
    #print(df_closest)
    #st.write('Here are the ten closest grades and their course outcomes:')
    #st.write(df_closest)

    fig1 = plot_prediction_1D(lr_avg, new_avg, X['Weighted Avg'], y)

    right_column.write('Here is a chart showing the estimate of your probability of passing the course.')
    right_column.write(fig1)
    

elif option == opt2:
        
    st.write('**Please enter quiz average and midterm 1 score.**')
    new_mid1 = st.number_input('**Please input your midterm 1 score as a decimal (e.g., 0.71, not 71).**', min_value=0.00, max_value=1.00, value='min')
    new_quiz = st.number_input('**Please input your quiz average as a decimal (e.g., 0.71, not 71).**', min_value=0.00, max_value=1.00, value='min')
    #new_avg=0.6
    df_closest = find_closest_2D(new_mid1, new_quiz, 10, X[['Quiz Avg', 'Midterm1']], y)
    df_closest.index = np.arange(1, len(df_closest)+1)

    con = st.container(border=True)
    con.write(f'#### The logistic regression model estimates that you have a {lr_some.predict_proba([[new_mid1, new_quiz]])[0,1]*100:.03}% probability of passing the course.')

    left_column, right_column = st.columns(2)
    # You can use a column just like st.sidebar:
    left_column.write('Here are the ten closest grades and their course outcomes:')
    left_column.dataframe(df_closest, use_container_width=True)
    left_column.write(f'Among the students with the 10 closest scores to yours, {np.sum([x==yes_str for x in df_closest.Passed])} passed the course.')
    #print(df_closest)
    #st.write('Here are the ten closest grades and their course outcomes:')
    #st.write(df_closest)

    fig1 = plot_prediction_2D(lr_some, new_mid1, new_quiz, X[['Quiz Avg', 'Midterm1']], y)

    right_column.write('Here is a figure showing your (midterm 1, quiz average) pair along with 50 closest (midterm 1, quiz average) pairs color coded by course outcome.')
    right_column.write(fig1)