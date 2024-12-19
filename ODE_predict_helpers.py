import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings('ignore')

def model_train(X, y, features_list, balance = None):
    '''
    Define and train a logistic regression model to predict course outcome using the features in features_list 
    Arguments:
        X - dataframe of features
        y - series of target variable
        features_list - list of column names from X to be used to train the logistic regression classifier
        balance - None or 'balanced' -- used in the class_weight argument in LogisticRegression.
    Outputs:
        Trained classifier, X_test, y_test
    '''
    #X_train, X_test, y_train, y_test = train_test_split(X[features_list], y, random_state=0)


    lr = LogisticRegression(class_weight=balance,random_state=0)
    #params = {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}#, 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]} 

    #clf = GridSearchCV(lr, params, scoring = 'accuracy')
    lr.fit(X[features_list], y)

    return lr

    
def find_closest_1D(new_weighted_avg, num_closest, X_weighted_avg, y):

    '''
    Returns num_closest students from the ODE grade dataset with the closest weighted course averages to new_weighted_avg

    Arguments:
        new_weighted_average - weighted course average of a new student (for prediction)
        num_closest - the number of the closest datapoints to return
        X_weighted_avg - 'Weighted Avg' column from the X dataframe
        y - target vector of 0 and 1's corresponding to fail and pass
    Returns:
        Dataframe of closest datapoints and pass/fail data
    '''

    k_closest_idx = np.argpartition(np.abs(X_weighted_avg-new_weighted_avg), num_closest)[:num_closest]

    df_closest = pd.DataFrame()
    df_closest.insert(0, column = 'Weighted Avg', value = X_weighted_avg[k_closest_idx])
    df_closest.insert(1, column = 'Passed', value = ['Yes' if x==1 else 'No' for x in y[k_closest_idx]])
    df_closest.sort_values(by = ['Weighted Avg'], axis=0, inplace=True)
    return df_closest.reset_index(drop=True)



def find_closest_2D(new_midterm1, new_quiz_avg, num_closest, X_some, y):

    '''
    Returns num_closest students from the ODE grade dataset with the closest (in Euclidean distance) weighted course averages to new_weighted_avg

    Arguments:
        new_midterm1 - midterm 1 score of a new student (for prediction)
        new_quiz_average -  quiz average of a new student (for prediction)
        num_closest - the number of the closest datapoints to return
        X_weighted_avg - 'Weighted Avg' column from the X dataframe
        y - target vector of 0 and 1's corresponding to fail and pass
    Returns:
        Dataframe of closest datapoints and pass/fail data
    '''

    k_closest_idx = np.argpartition(np.abs(X_some['Quiz Avg']-new_quiz_avg)**2 + (np.abs(X_some['Midterm1']-new_midterm1)**2), num_closest)[:num_closest]

    df_closest = pd.DataFrame()
    df_closest.insert(0, column = 'Midterm1', value = X_some['Midterm1'].iloc[k_closest_idx])
    df_closest.insert(1, column = 'Quiz Avg', value = X_some['Quiz Avg'].iloc[k_closest_idx])
    df_closest.insert(2, column = 'Passed', value = ['Yes' if x==1 else 'No' for x in y[k_closest_idx]])
    df_closest.insert(3, column = 'Dist', value = np.abs(X_some['Quiz Avg']-new_quiz_avg)**2 + (np.abs(X_some['Midterm1']-new_midterm1)**2))
    df_closest.sort_values(by = ['Dist'], axis=0, inplace=True)
    return df_closest.drop('Dist', axis=1).reset_index(drop=True)



def plot_prediction_1D(model, new_weighted_avg, X_weighted_avg, y):
    '''
    Returns figure of passing probabilities and the outcomes of the 50 closest past students

    Arguments:
        model - trained model to use
        new_weighted_average -  weight course average of a new student (for prediction)
        X_weighted_avg - 'Weighted Avg' column from the X dataframe
        y - target vector of 0 and 1's corresponding to fail and pass
    Returns:
        Dataframe of closest datapoints and pass/fail data
    '''
    X_range = np.linspace(0,1,50).reshape(-1,1)
    y_probs_range = model.predict_proba(X_range)[:,1]
    
    closest = find_closest_1D(new_weighted_avg, 10, X_weighted_avg, y)
    closest_probs = model.predict_proba(closest[['Weighted Avg']])[:,1]
    closest_passed = (closest['Passed'].values=='Yes')*1
    new_pass_proba = model.predict_proba([[new_weighted_avg]])[0,1]
    
    colormap = np.array(['r', 'g'])

    fig = plt.figure(figsize=(4,3))
    plt.plot(X_range,y_probs_range,linewidth=3)
    #plt.scatter(closest['Weighted Avg'], closest_passed, c=colormap[closest_passed], alpha=0.5, s=50)
    plt.plot(new_weighted_avg, new_pass_proba, 'o', c='tab:red', markersize=10)
    plt.plot([new_weighted_avg, new_weighted_avg], [0, new_pass_proba], '--', c='tab:red')
    plt.plot([0, new_weighted_avg], [new_pass_proba, new_pass_proba], '--', c='tab:red')
    plt.xlabel('Weighted Course Average') 
    plt.ylabel('Pass Probability')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    return fig


def plot_prediction_2D(model, new_midterm1, new_quiz_avg, X_some, y):
    '''
    Returns figure of passing probabilities and the outcomes of the 50 closest past students

    Arguments:
        model - trained model to use
        new_midterm1 - midterm 1 score of a new student (for prediction)
        new_quiz_average -  quiz average of a new student (for prediction)
        X_weighted_avg - 'Weighted Avg' column from the X dataframe
        y - target vector of 0 and 1's corresponding to fail and pass
    Returns:
        Dataframe of closest datapoints and pass/fail data
    '''
    N = 200
    x = np.linspace(0, 1, N)
    x1, x2 = np.meshgrid(x, x)
    P_prob = np.zeros((N,N))

    closest = find_closest_2D(new_midterm1, new_quiz_avg, 50, X_some, y)
    closest_probs = model.predict_proba(closest[['Midterm1', 'Quiz Avg']])[:,1]
    closest_passed = (closest['Passed'].values=='Yes')*1

    new_pass_proba = model.predict_proba([[new_midterm1, new_quiz_avg]])[0,1]

    colormap = np.array(['r', 'g'])

    for i in range(N):
        for j in range(N):
            #dat = pd.DataFrame(data = {'Midterm1': [x[j]], 'Quiz Avg': [x[i]]})
            P_prob[i, j] = model.predict_proba([[x[j],x[i]]])[0,1]

    #plt.subsplots()
    fig = plt.figure(figsize=(6,4))
    im = plt.imshow(P_prob, origin = 'lower', extent = [0, 1, 0, 1], alpha=0.5)
    colormap1 = ListedColormap(['r', 'g'])
    colormap2 = ListedColormap(['b'])
    #scatter = plt.scatter(X['Quiz Avg'], X['Midterm1'],c=y.values,cmap=colormap)
    scatter1 = plt.scatter(closest['Midterm1'], closest['Quiz Avg'], c=closest_passed, cmap=colormap1, alpha=1, s=30)
    scatter2 = plt.scatter(new_midterm1, new_quiz_avg, c=[0], cmap=colormap2, s=150)
    plt.xlabel('Midterm 1 Score', fontsize=14)
    plt.ylabel('Quiz Average', fontsize=14)
    cbar = plt.colorbar(im, alpha=0.5)
    cbar.set_label('Pass probability', fontsize=14)
    #handles, labels = .legend_elements()
    plt.legend(handles=scatter1.legend_elements()[0][::-1] + scatter2.legend_elements()[0], labels=['Pass','Fail','Your scores'], fontsize=9, loc='lower center', ncol=3)
    
    return fig

