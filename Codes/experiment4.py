
import glob
#import matlab.engine 
from scipy import stats

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
#from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from treeinterpreter import treeinterpreter as ti
from randomf import random_forest
from sklearn.metrics import f1_score

#from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


"""
This code comapre the roc accuracies between the parcellated and unparcellated striatal volume features.
"""

out_path = "../Results/Training/"
regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]
feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf_disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]


def get_center_data(df,session):
  """This method reads participant.tsv files and add them to csv files from diffparc

  inputs: session = Session that you want to sort the file. Eg: Baseline"""

  #Reading csv files obtained from diffparc_sumo
  center = pd.read_csv("../Data/demographic/Center-Subject_List.csv")

  center = center.rename(columns={'PATNO': 'subj'})
  subj_id = list(center["subj"])
  center_id = list(center["CNO"])

  for i,IDs in enumerate(subj_id):
    subj_id[i]= 'sub-'+str(IDs)+'_ses-'+session

  center.drop('subj', axis = 1, inplace = True)
  center['subj'] = subj_id

  df = pd.merge(df, center, on='subj')

  ID_list = df["CNO"]
  #print(ID_list)
  empty = []
  new_list = []
  k =1;

  for i,IDs in enumerate(ID_list):
    
    #print("IDs:",IDs)
    if IDs not in empty:
      empty.append(IDs)
      new_list.append(k)
      #print("epmty",empty)
      #print("new",new_list)
      k=k+1
      #print("k:",k)

    else:
      
      #empty.append(ID_list[i])
      ind = empty.index(IDs)
      #ID_list[i] = ind
      new_list.append(ind+1)
      #print("ind:",ind+1)

  df.drop("CNO", axis = 1, inplace = True)
  df["CNO"] = new_list

  return df



def get_new_header(feature_list):
  #The reads a list of numbers corresponding to features and concat that number to region names
  #so that it creates a new heasder list to select features from the df.

  new_header = []
  for i in feature_list:
    for rois in regions:
      new_header.append(rois+'_'+str(i))

  return new_header

def just_split(df):
    #This function is just splitting the data into train_x and train_y. No test set

    header = list(df)

    feature_headers = [x for x in header if x not in ['subj', 'group_ID','DOMSIDE','MDS-UPDRS_Total']]

    target_header = 'group_ID'

    train_x = df[feature_headers] 
    train_y = df[target_header]
    #subject_list = df['subj']

    #Uncomment to select specific features
    new_header = get_new_header([10])
    train_x = train_x[new_header]

    return train_x,train_y




def get_cv_results(train_x,train_y,folds=5):
  #This function takes a training data set and a classifier and run cross validation.
  #Within each split, it calculates accuracy, roc curve and auc and gives the mean values and mean roc plot.
  #Also it calculates the feature importance and save them.
  #train_x: panda df with features
  #train_y: panda df with classes
  #clf: a classifier 
  #subject_list: panda df with subject names
  #folds: Number of folds (int). Default = 5

  #Turning df into arrays
  X = np.array(train_x)
  y = np.array(train_y)
  #subj = np.array(subject_list)

  columns = train_x.columns

  kf = KFold(n_splits=folds,shuffle=True)

  print("Cross validation info:", kf)

  models = []

  tprs = []
  auc = []
  apc = []
  test_acu = []
  f1 = []
  base_fpr = np.linspace(0, 1, 101)

  T_P = 0
  T_N = 0
  F_P = 0
  F_N = 0

  
  #iterating through splits, fitting the model and calculate accuracies and also plots roc curves
  for i, (train, test) in enumerate(kf.split(X)):

    clf = random_forest(X[train], y[train])
    model = clf.fit(X[train], y[train])
    models.append(model)
    y_score = model.predict_proba(X[test])[:,1]

    #Getting roc values
    fpr, tpr, _ = roc_curve(y[test], y_score)
    #fpr, tpr, _ = precision_recall_curve(y[test], y_score, pos_label = 1)
    roc_value = roc_auc_score(y[test], y_score, average = "micro")
    f1_value = f1_score(y[test], np.around(y_score),average='micro')
    average_precision = average_precision_score(y[test], y_score, pos_label = 1)
    print("average_precision_score",average_precision)

    f1.append(f1_value)
    auc.append(roc_value)
    apc.append(average_precision)

    #plt.plot(fpr, tpr, 'b', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

    #Calculating test accuracy
    predictions = model.predict(X[test])
    test_accuracy = accuracy_score(y[test], predictions)
    test_acu.append(test_accuracy)
    print("Test Accuracy  :: ", test_accuracy)
    #print(" Confusion matrix ", confusion_matrix(y[test], predictions))
    T_P = T_P + confusion_matrix(y[test], predictions)[0,0]
    F_P = F_P + confusion_matrix(y[test], predictions)[0,1]
    F_N = F_N + confusion_matrix(y[test], predictions)[1,0]
    T_N = T_N + confusion_matrix(y[test], predictions)[1,1]


    #get_importance(model,columns,X[test],y[test],subj[test])

    #Confusion matrix (needed to be updated)
    #print(" Confusion matrix ", confusion_matrix(test_y, y_score))

  
  tprs = np.array(tprs)
  mean_tprs = tprs.mean(axis=0)
  std = tprs.std(axis=0)
  mean_f1 = np.mean(f1,axis=0)

  tprs_upper = np.minimum(mean_tprs + std, 1)
  tprs_lower = mean_tprs - std

  mean_auc = np.mean(auc,axis=0)
  std_auc = np.std(auc,axis=0)
  print("mean area under the curve: {0: 0.3f} (+/- {1: 0.3f})"\
            .format(mean_auc, std_auc / 2))

  mean_apc = np.mean(apc,axis=0)
  std_apc = np.std(apc,axis=0)
  print("mean area under the precision curve: {0: 0.3f} (+/- {1: 0.3f})"\
            .format(mean_apc, std_apc / 2))

  print("mean F1 value:",mean_f1)

  conf_matrix = [T_P,F_P,F_N,T_N]
  #np.savetxt(out_path+'/confusion_matrix_early_PD.csv', conf_matrix, fmt='%s', delimiter=',')


  mean_test_acu = np.mean(test_acu,axis=0)
  std_test_acu = np.std(test_acu,axis=0)
  print("mean test accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
            .format(mean_test_acu, std_test_acu / 2))

  #comment these if u are running the selected features


  return models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu

def parcellated_vol():
  #Calculating accuraciea for parcellated volumes

  df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_UPDRS.csv')

  train_x,train_y = just_split(df)

  models, base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = get_cv_results(train_x,train_y)

  return models, base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu

def unparcellated_vol():
  #Calculating accuracies for unparcellated volumes

  df = pd.read_csv("../Data/PD/volumes_seed-CIT168striatum_targets-cortical_anatomical_space-T1w_1.csv")
  df = df.dropna()

  selected = pd.read_csv("../Data/combat/all/combined_BL/Combined_with_UPDRS.csv")

  selected = selected[selected['group_ID'] == 1]

  selected_subjecs = list(selected['subj'])

  PD = df[df['subj'].isin(selected_subjecs)]
  #PD = PD.drop(['subj'], axis=1)

  ID = np.ones(PD.shape[0], dtype=int)
  PD['group_ID']= ID
  #print(PD)

  df = pd.read_csv("../Data/CT/volumes_seed-CIT168striatum_targets-cortical_anatomical_space-T1w.csv")
  df = df.dropna()

  selected = pd.read_csv("../Data/combat/all/combined_BL/Combined_with_UPDRS.csv")

  selected = selected[selected['group_ID'] == 0]

  selected_subjecs = list(selected['subj'])

  CT = df[df['subj'].isin(selected_subjecs)]
  #CT = CT.drop(['subj'], axis=1)

  ID = np.zeros(CT.shape[0], dtype=int)
  CT['group_ID']= ID

  Data = pd.concat([CT,PD])


  Data = get_center_data(Data,"Baseline")


  Data.to_csv('../Data/combat/volume/just_volume.csv', sep=',', index=False)
  harmonized_file= '../Data/combat/volume/harmonized_volume.csv'

  #if not os.path.exists(harmonized_file):
  #  eng = matlab.engine.start_matlab()
  #  eng.harmonize_volume(nargout=0)
  #  eng.quit()

  Data = pd.read_csv(harmonized_file)
  y=Data.group_ID
  X=Data.drop(['group_ID','subj','CNO'], axis=1)

  models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = get_cv_results(X,y)

  return models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu


def get_volume_roc():
  #Get roc curves for volume feature comparision between parcellated and unparcellated features

  plt.figure(figsize=(5, 5))

  models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = unparcellated_vol()
  plt.plot(base_fpr, mean_tprs, 'b', label=r'unparcellated volumes (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)

  #models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = parcellated_vol()
  #plt.plot(base_fpr, mean_tprs, 'g', label=r'parcellated volumes (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)


  plt.plot([0, 1], [0, 1],'r--',label='Chance')
  plt.xlim([-0.01, 1.01])
  plt.ylim([-0.01, 1.01])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.axes().set_aspect('equal', 'datalim')
  plt.legend(loc="lower right")
  plt.savefig(out_path+'parce_vs_non_PD_vs_CT.pdf', bbox_inches='tight')
  plt.show()


get_volume_roc()


