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




out_path = "../Results/Training/"

def make_out_dir(out_path):

	#Make subdirectories to save files. If old folder exist, it will delete it.
	filename = out_path+"feature_importance/"
	if os.path.exists(os.path.dirname(filename)):

		shutil.rmtree(filename)
	os.makedirs(os.path.dirname(filename))

make_out_dir(out_path)

#UPDRS, title = True, 'early_PD'
UPDRS, title = False, 'PD'

regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]
feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf_disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]

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

    if UPDRS == False:
        target_header = 'group_ID'
    if UPDRS == True :
        target_header = 'MDS-UPDRS_Total'

    train_x = df[feature_headers] 
    train_y = df[target_header]
    subject_list = df['subj']

    #Uncomment to select specific features
    #new_header = get_new_header([9])
    #train_x = train_x[new_header]

    return train_x,train_y,subject_list

def test_mode():

	if UPDRS == False:
		df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_UPDRS.csv')
	if UPDRS == True:
		df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_earlyPD.csv')

	train_x,train_y,subject_list = just_split(df)

	#clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #        max_depth=20, max_features='auto', max_leaf_nodes=None,
    #        min_impurity_decrease=0.0, min_impurity_split=None,
    #        min_samples_leaf=1, min_samples_split=2,
    #        min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
    #        oob_score=False, verbose=0, warm_start=False).fit(train_x,train_y)

	models, base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = get_cv_results(train_x,train_y,subject_list)

	return models

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


def get_importance(model,columns,test_x,test_y,subj):
	#This function calculates the feature importance for a give test set and plots and saves values only for true PD detects.
	#model : Trained RF model
	#columns: List of column names
	#test_x: Array of test data
	#test_y: Array of test classes
	#subj: Array of subject names of the test set
	#returns: Just plots the feature importance and save as a csv.


	feature_headers = get_header_names(columns)
	prediction, bias, contributions = ti.predict(model,test_x)

	#This for loop goes through the test subjects and plots the feature importance for only PD subjects and save
	for i in range(len(test_x)):

		ind = np.argmax(prediction[i])
		#print(ind)
		#print(test_y[i])
		#If the prediction is a PD and it is true, then save importance
		if ind == 1 and test_y[i] == 1.0:
			contribution = (contributions[i])[:,ind]
			importance = pd.DataFrame({'Variable':feature_headers,
	              'Contribution':contribution}).sort_values('Contribution', ascending=False)
			#sns.barplot(x="Contribution", y="Variable", data=importance)

			#print(importance)
			importance.to_csv(out_path+"feature_importance/"+subj[i]+"_feature_importance_true_P.csv", sep=',', index=False)

			#print(sorted(zip(contribution,feature_headers), key=lambda x: -abs(x[0])))
	                             
			#plt.show()

		if ind == 0 and test_y[i] == 0.0:
			contribution = (contributions[i])[:,ind]
			importance = pd.DataFrame({'Variable':feature_headers,
	              'Contribution':contribution}).sort_values('Contribution', ascending=False)
			#sns.barplot(x="Contribution", y="Variable", data=importance)

			#print(importance)
			importance.to_csv(out_path+"feature_importance/"+subj[i]+"_feature_importance_true_N.csv", sep=',', index=False)

		if ind == 1 and test_y[i] == 0.0:
			contribution = (contributions[i])[:,ind]
			importance = pd.DataFrame({'Variable':feature_headers,
	              'Contribution':contribution}).sort_values('Contribution', ascending=False)
			#sns.barplot(x="Contribution", y="Variable", data=importance)

			#print(importance)
			importance.to_csv(out_path+"feature_importance/"+subj[i]+"_feature_importance_F_N.csv", sep=',', index=False)

		if ind == 0 and test_y[i] == 1.0:
			contribution = (contributions[i])[:,ind]
			importance = pd.DataFrame({'Variable':feature_headers,
	              'Contribution':contribution}).sort_values('Contribution', ascending=False)
			#sns.barplot(x="Contribution", y="Variable", data=importance)

			#print(importance)
			importance.to_csv(out_path+"feature_importance/"+subj[i]+"_feature_importance_F_P.csv", sep=',', index=False)
		"""contribution = (contributions[i])[:,ind]
		importance = pd.DataFrame({'Variable':feature_headers,
              'Contribution':contribution}).sort_values('Contribution', ascending=False)
		#sns.barplot(x="Contribution", y="Variable", data=importance)

		#print(importance)
		importance.to_csv(out_path+"feature_importance/"+subj[i]+"_feature_importance.csv", sep=',', index=False)

		#print(sorted(zip(contribution,feature_headers), key=lambda x: -abs(x[0])))
                             
		#plt.show()"""
			

def get_cv_results(train_x,train_y,subject_list,folds=5):
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
	subj = np.array(subject_list)

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

	#plt.figure(figsize=(5, 5))

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
	np.savetxt(out_path+'/confusion_matrix_early_PD.csv', conf_matrix, fmt='%s', delimiter=',')


	mean_test_acu = np.mean(test_acu,axis=0)
	std_test_acu = np.std(test_acu,axis=0)
	print("mean test accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
	          .format(mean_test_acu, std_test_acu / 2))

	#comment these if u are running the selected features
	"""plt.plot(base_fpr, mean_tprs, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)
	plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label=r'$\pm$ 1 std. dev.')

	plt.plot([0, 1], [0, 1],'r--',label='Chance')
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.axes().set_aspect('equal', 'datalim')
	plt.legend(loc="lower right")
	plt.savefig(out_path+'mean_roc_curve_PD_vs_CT.pdf', bbox_inches='tight')
	plt.show()"""

	return models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu

def get_roc_for_selected_fetres(df):

	accuracy_array = []
	plt.figure(figsize=(5, 5))

	train_x_original,train_y,subject_list = just_split(df)

	SASD_header = get_new_header([8,9])
	train_x = train_x_original[SASD_header]

	models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = get_cv_results(train_x,train_y,subject_list)
	
	print(base_fpr)
	
	print(mean_tprs)
	
	plt.plot(base_fpr, mean_tprs, 'b',label='SA & SD (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)

	accuracy_array.append(["SA_SD", mean_auc,std_auc, mean_test_acu, std_test_acu])

	FAMD_header = get_new_header([3,6,7])
	train_x = train_x_original[FAMD_header]

	models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = get_cv_results(train_x,train_y,subject_list)
	plt.plot(base_fpr, mean_tprs, 'r',label='FA,MD & connectivity (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)

	accuracy_array.append(["DTI", mean_auc,std_auc, mean_test_acu, std_test_acu])

	volume_header = get_new_header([10])
	train_x = train_x_original[volume_header]

	models,base_fpr, mean_tprs,mean_auc, std_auc, mean_test_acu, std_test_acu = get_cv_results(train_x,train_y,subject_list)
	plt.plot(base_fpr, mean_tprs, 'g',label='volume (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)

	accuracy_array.append(["Volume", mean_auc,std_auc, mean_test_acu, std_test_acu])

	print(accuracy_array)

	plt.plot([0, 1], [0, 1],'k--',label='Chance')
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.axes().set_aspect('equal', 'datalim')
	plt.legend(loc="lower right")
	plt.savefig(out_path+'selected_features_mean_roc_curve_'+title+'.pdf', bbox_inches='tight')

	plt.show()







if UPDRS == False:
	df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_UPDRS.csv')
if UPDRS == True:
	df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_earlyPD.csv')
get_roc_for_selected_fetres(df)

#models = test_mode()

#random_forest_validation(models)



