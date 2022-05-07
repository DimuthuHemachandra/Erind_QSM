
import glob
import shutil
# pandas for data loading, manipulation etc.
import pandas as pd

# numeric functions
import numpy as np
from scipy import stats
from math import ceil

# plotting
import matplotlib.pyplot as plt
import seaborn as sns


# modelling
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from roc import get_cv_results



#rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 5, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
#plt.rcParams.update(**rc)
#sns.set(rc=rc)



out_path = "../Results/Training/"

regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]

feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf_disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]

#UPDRS = False
UPDRS = True

# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)    
    n = len(y_pred)   
    
    return np.sqrt(sum_sq/n)

# scorer to be used in sklearn model fitting
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# places to store optimal models and scores
opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])

# no. k-fold splits
splits=5
# no. k-fold iterations
repeats=5


def train_model(model, param_grid=[], X=[], y=[], 
                splits=5, repeats=5):


    
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    
    # perform a grid search if param_grid given
    if len(param_grid)>0:
        # setup grid search parameters
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring=rmse_scorer,
                               verbose=1, return_train_score=True)

        # search the grid
        gsearch.fit(X,y)

        # extract best model from the grid
        model = gsearch.best_estimator_        
        best_idx = gsearch.best_index_

        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)       
        cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
        cv_std = grid_results.loc[best_idx,'std_test_score']

        print(gsearch.best_params_)


        #plot.grid_search(gsearch.grid_scores_, change='n_estimators', kind='bar')
        #plt.show()

    # no grid search, just cross-val score for given model   
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y, scoring=rmse_scorer, cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)
    
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)
    
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=',model.score(X,y))
    print('rmse=',rmse(y, y_pred))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    
    # residual plots
    """y_pred = pd.Series(y_pred,index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid)/std_resid    
    n_outliers = sum(abs(z)>3)

    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y_pred');
    plt.title('corr = {:.3f}'.format(np.corrcoef(y,y_pred)[0][1]))
    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred');
    plt.title('std resid = {:.3f}'.format(std_resid))
    
    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))

    plt.show()"""

    return model, cv_score, grid_results






def get_header_names(header_list):
	#This function replaces the numbers in the header names and replace it with feature names

	new_list = []
	for name in header_list:
		splitted = name.split('_')
		number = splitted[-1]
	
		for i in range (1,11):
			if int(number) == i:

				newstr = name.replace(number, feature_names[i-1])
				new_list.append(newstr)
			
	return new_list



def split_dataset(df):
    #Splits the dataset into training and testing

    header = list(df)

    feature_headers = [x for x in header if x not in ['subj', 'group_ID','DOMSIDE','MDS-UPDRS_Total']]

    if UPDRS == False:
        target_header = 'group_ID'
    if UPDRS == True :
        target_header = 'MDS-UPDRS_Total'

    train_x, test_x, train_y, test_y = train_test_split(df[feature_headers], df[target_header],train_size=0.7)

    # Train and Test dataset size details
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)


    return train_x, test_x, train_y, test_y

def just_split(df):
    #This function is just splitting the data into train_x and train_y. No test set

    header = list(df)

    feature_headers = [x for x in header if x not in ['subj', 'group_ID','DOMSIDE','MDS-UPDRS_Total']]

    if UPDRS == False:
        target_header = 'group_ID'
    if UPDRS == True :
        target_header = 'MDS-UPDRS_Total'

    train_x = df[feature_headers] 
    train_y = df[target_header]

    return train_x,train_y



def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


def get_new_header(df,feature_list):
    #The reads a list of numbers corresponding to features and concat that number to region names
    #so that it creates a new heasder list to select features from the df.

    header = list(df)

    new_header = []
    for i in feature_list:
        for rois in regions:
            new_header.append(rois+'_'+str(i))

    return new_header


def random_forest(score_models):

    #train_x, test_x, train_y, test_y = split_dataset(df)
    train_x, train_y =just_split(df)
    subject_list = df['subj']

    #train_x_original, test_x_original, train_y, test_y = split_dataset(df)

    #Selecting a specific feature/s
    #####################################
    #new_header = get_new_header(df,[9])

    #train_x = train_x_original[new_header]
    #test_x = test_x_original[new_header]
    ######################################


    model = 'RandomForest'
    opt_models[model] = RandomForestClassifier(random_state=42)

    param_grid = {'n_estimators':[40,60,100,150,200,300],
              'min_samples_split':[2,4,6],'max_features':['auto', 'sqrt'],'max_depth': [20, 40, 60, 80, 100, None]}

    clf, cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, 
                                                  splits=5, repeats=1, X = train_x, y = train_y)

    cv_score.name = model
    score_models = score_models.append(cv_score)


    if UPDRS == False:
        grid_results.to_csv(out_path+"grid_results.csv", sep=',', index=False)
    if UPDRS == True:
        grid_results.to_csv(out_path+"grid_results_with_UPDRS1.csv", sep=',', index=False)


    train_x,train_y = just_split(df)

    #clf.fit(train_x,train_y)

    #clf, cv_score, grid_results = train_model(clf, param_grid=[], splits=5, repeats=1, X = train_x, y = train_y)

    rkfold = RepeatedKFold(n_splits=5, n_repeats=1)
    scores = cross_val_score(clf,train_x,train_y,cv = rkfold)

    print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
          .format(scores.mean(), scores.std() / 2))

    get_cv_results(train_x,train_y,clf,subject_list)




if UPDRS == False:
    df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_UPDRS.csv')
if UPDRS == True:
    df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_earlyPD.csv')

random_forest(score_models)







