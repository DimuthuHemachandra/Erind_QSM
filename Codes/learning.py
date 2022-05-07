import numpy as np 
import pandas as pd
import glob
import shutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PD_features import get_side_affected,get_PD_medication
from sklearn import manifold
from time import time
from matplotlib import offsetbox
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_boston

from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.tree import export_graphviz

from treeinterpreter import treeinterpreter as ti



#rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 5, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
#plt.rcParams.update(**rc)
#sns.set(rc=rc)

#To make white plots
"""params = {"ytick.color" : "w",
      "xtick.color" : "w",
      "axes.labelcolor" : "w",
      "axes.edgecolor" : "w"}
plt.rcParams.update(params)"""



out_path = "../Results/tsne/"

regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]

feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf_disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]

UPDRS = False
#UPDRS = True


if UPDRS == False:
	df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_UPDRS.csv')
	#df=pd.read_csv('../Results/PCA/all_combined.csv')
if UPDRS == True:
	df=pd.read_csv('../Data/combat/all/combined_BL/Combined_with_earlyPD.csv')



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
	#This function splits a dataset into training and testing.

	header = list(df)

	#Removing unwanted columns
	feature_headers = [x for x in header if x not in ['subj', 'group_ID','DOMSIDE','MDS-UPDRS_Total']]

	#Selecting a column to be used as labels
	if UPDRS == False:
	    target_header = 'group_ID'
	if UPDRS == True :
	    target_header = 'MDS-UPDRS_Total'
	

	train_x, test_x, train_y, test_y = train_test_split(df[feature_headers], df[target_header],
	                                                    train_size=0.7)

	# Train and Test dataset size details
	print("Train_x Shape :: ", train_x.shape)
	print("Train_y Shape :: ", train_y.shape)
	print("Test_x Shape :: ", test_x.shape)
	print("Test_y Shape :: ", test_y.shape)


	return train_x, test_x, train_y, test_y


def do_TSNE(df,dataset_name):
	#t-SNE for clustering

	print("Computing t-SNE embedding")

	if UPDRS == True:
		y = np.array(list(df["MDS-UPDRS_Total"]))
		df.drop(['subj', 'group_ID','MDS-UPDRS_Total'], axis=1, inplace=True)
		colors = ['navy', 'turquoise','darkorange']
		n_groups = [0,1]
		target_names = ["CT","Early_PD"]
	else:
		y = np.array(list(df["group_ID"]))
		df.drop(['subj', 'group_ID'], axis=1, inplace=True)
		colors = ['navy', 'turquoise']
		n_groups = [0,1]
		target_names = ["CT","PD"]

	
	#Uncomment this to get PCA for selected features.
	#new_header = get_new_header(df,[9])
	#df = df[new_header]

	X = np.array(df.values)


	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	t0 = time()
	X_tsne = tsne.fit_transform(X)

	print(X_tsne)

	plt.figure()
	lw = 2
	for color, i, target_name in zip(colors, n_groups, target_names):
	    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], color=color, alpha=.8, lw=lw,
	                label=target_name)


	plt.legend(loc='best', shadow=False, scatterpoints=1)
	#plt.title('TSNE of PPMI data with' + dataset_name)

	plt.xlabel('C1')
	plt.ylabel('C2')
	plt.savefig(out_path+'t_SNE_all.eps', bbox_inches='tight')
	plt.show()


def get_lasso_and_elasticnet(df):
	
	train_x, test_x, train_y, test_y = split_dataset(df)

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)


	alpha = 0.001
	lasso = Lasso(alpha=alpha)

	y_pred_lasso = lasso.fit(train_x, train_y).predict(test_x)
	r2_score_lasso = r2_score(test_y, y_pred_lasso)
	print(lasso)
	print("r^2 on test data : %f" % r2_score_lasso)

	# #############################################################################
	# ElasticNet

	
	enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

	y_pred_enet = enet.fit(train_x, train_y).predict(test_x)
	r2_score_enet = r2_score(test_y, y_pred_enet)
	print(enet)
	print("r^2 on test data : %f" % r2_score_enet)

	plt.plot(enet.coef_, color='lightgreen', linewidth=2,
	         label='Elastic net coefficients')
	plt.plot(lasso.coef_, color='gold', linewidth=2,label='Lasso coefficients')
	plt.legend(loc='best')
	plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
	          % (r2_score_lasso, r2_score_enet))

	plt.xlabel('Features')
	plt.savefig(out_path+'lasso_and_elasticnet.eps', bbox_inches='tight')
	#plt.show()

	predictions = enet.predict(test_x)

	en_coefs = enet.coef_
	print(np.size(en_coefs))

	header = list(df)

	feature_headers = [x for x in header if x not in ['subj', 'group_ID','DOMSIDE','MDS-UPDRS_Total']]

	feature_headers = get_header_names(feature_headers)

	en_coefs = pd.Series(en_coefs,feature_headers)
	plt.figure()
	en_coefs[en_coefs.abs()>0.02].sort_values().plot.barh()
	plt.tight_layout()
	plt.savefig(out_path+'elasticnet_selected_features.eps', bbox_inches='tight')
	plt.show()

	# Train and Test Accuracy

	print("Train Accuracy :: ", enet.score(train_x, train_y))
	#print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
	#print(" Confusion matrix ", confusion_matrix(test_y, predictions))

##################################################################################

def score(x1,x2):
	return mean_squared_error(x1,x2)
# defining feature importance function based on above logic
def feat_imp(m, x, y, small_good = True): 

	score_list = {} 
	score_list['original'] = score(m.predict(x.values), y) 
	imp = {} 
	for i in range(len(x.columns)): 
	    rand_idx = np.random.permutation(len(x)) # randomization
	    new_coli = x.values[rand_idx, i] 
	    new_x = x.copy()            
	    new_x[x.columns[i]] = new_coli 
	    score_list[x.columns[i]] = score(m.predict(new_x.values), y) 
	    imp[x.columns[i]] = score_list['original'] - score_list[x.columns[i]] # comparison with benchmark
	if small_good: 
	     return sorted(imp.items(), key=lambda x: x[1]) 


##################################################################################


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """

    #np.random.seed(1234)
    #UPDRS20
    #clf = RandomForestClassifier(n_estimators = 40, min_samples_split = 2, max_features = 'sqrt', max_depth = 80)
    #UPDRS30
    #clf = RandomForestClassifier(n_estimators = 40, min_samples_split = 6, max_features = 'auto', max_depth = 40)
    #clf = RandomForestClassifier()
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=20, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
            oob_score=False, verbose=0, warm_start=False)
    clf.fit(features, target)
    return clf


def do_random_forest(df):


	train_x, test_x, train_y, test_y = split_dataset(df)

	# Create random forest classifier instance
	trained_model = random_forest_classifier(train_x, train_y)
	print("Trained model :: ", trained_model)

	predictions = trained_model.predict(test_x)

	for i in range(0, 10):
	    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

	# Train and Test Accuracy
	print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
	print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
	print(" Confusion matrix ", confusion_matrix(test_y, predictions))


	y_predicted = trained_model.predict_proba(test_x)[:,1]
	print(y_predicted)
	test_y1 = list(test_y)
	print(test_y1)
	#dic = {1:0, 2:1}
	#y_test = np.array([dic.get(n, n) for n in test_y1])
	#print(y_test)
	y_test = test_y

	roc_value = roc_curve(y_test, y_predicted)
	plt.plot(roc_value[0],roc_value[1])
	plt.xlabel("False positive rate (1- Specificity)")
	plt.ylabel("True positive rate (Sensitivity)")
	plt.title("ROC Curve")
	plt.savefig(out_path+'ROC_curve.eps', bbox_inches='tight')
	plt.show()

	roc_value = roc_auc_score(y_test, y_predicted)
	print(" ROC area :: ", roc_value)

	#importance = feat_imp(trained_model, train_x, train_y)

	#importance = trained_model.feature_importances_
	feature_headers = get_header_names(train_x.columns)
	importance = pd.DataFrame({'Variable':feature_headers,
              'Importance':trained_model.feature_importances_}).sort_values('Importance', ascending=False)

	#sns.set(rc={'figure.figsize':(15,20)})
	plt.figure()
	#sns.barplot(x="Importance", y="Variable", data=importance)
	#plt.tight_layout()
	#plt.savefig(out_path+'RF_feature_importance.eps', bbox_inches='tight')
	#plt.show()

	prediction, bias, contributions = ti.predict(trained_model,test_x)

	#This for loop goes through the first 5 test subjects and plots the feature importance for each
	print(np.shape(contributions[1]))
	for i in range(0,5):

		ind = np.argmax(prediction[i])
		print(ind)
		contribution = (contributions[i])[:,ind]
		importance = pd.DataFrame({'Variable':feature_headers,
              'Contribution':contribution}).sort_values('Contribution', ascending=False)
		sns.barplot(x="Contribution", y="Variable", data=importance)

	

		print(sorted(zip(contribution,feature_headers), key=lambda x: -abs(x[0])))
                             
		plt.show()

	  

	#Ths section is to print the Random Forest tree
	#######################################################################
	#estimator_nonlimited = trained_model.estimators_[5]
	#print(estimator_limited)

	"""export_graphviz(estimator_nonlimited, out_file='tree_limited.dot', feature_names = train_x.columns,
                class_names = ['CT', 'PD' ],
                rounded = True, proportion = False, precision = 2, filled = True)"""

	#!dot -Tpng tree_limited.dot -o tree_limited.png

	#######################################################################



def get_new_header(df,feature_list):
	#The reads a list of numbers corresponding to features and concat that number to region names
	#so that it creates a new heasder list to select features from the df.

	header = list(df)

	new_header = []
	for i in feature_list:
		for rois in regions:
			new_header.append(rois+'_'+str(i))

	return new_header





def random_forest_with_selected_features(df,feature_list,train_x_original, test_x_original, train_y, test_y):
	#Plots ROC curves for 4 different sets of features for comparison. 
	#This is just a quick fix. Not the final version.

	#train_x_original, test_x_original, train_y, test_y = split_dataset(df)

	#Selecting Ind. connectivity + Surface area + volume
	new_header = get_new_header(df,feature_list)

	train_x = train_x_original[new_header]
	test_x = test_x_original[new_header]


	# Create random forest classifier instance
	trained_model = random_forest_classifier(train_x, train_y)
	print("Trained model :: ", trained_model)

	predictions = trained_model.predict(test_x)

	for i in range(0, 10):
	    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

	# Train and Test Accuracy
	print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
	print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
	print(" Confusion matrix ", confusion_matrix(test_y, predictions))


	y_predicted = trained_model.predict_proba(test_x)[:,1]
	test_y1 = list(test_y)
	#dic = {1:0, 2:1}
	#y_test = np.array([dic.get(n, n) for n in test_y1])
	y_test = test_y

	roc_value = roc_curve(y_test, y_predicted)

	roc_auc_value = roc_auc_score(y_test, y_predicted)
	print(" Area under the curve :: ", roc_auc_value)

	#importance = feat_imp(trained_model, train_x, train_y)


	#print(importance)

	return roc_value



def get_roc_for_selected_fetres(df):

	train_x_original, test_x_original, train_y, test_y = split_dataset(df)

	roc_value = random_forest_with_selected_features(df,[8,9],train_x_original, test_x_original, train_y, test_y)
	plt.plot(roc_value[0],roc_value[1],'y',label = 'Surf_area + surf_disp')

	roc_value = random_forest_with_selected_features(df,[3,6],train_x_original, test_x_original, train_y, test_y)
	plt.plot(roc_value[0],roc_value[1],'orange',label = 'FA+MD')

	roc_value = random_forest_with_selected_features(df,[7],train_x_original, test_x_original, train_y, test_y)
	plt.plot(roc_value[0],roc_value[1],'red',label = 'Connectivity')

	roc_value = random_forest_with_selected_features(df,[10],train_x_original, test_x_original, train_y, test_y)
	plt.plot(roc_value[0],roc_value[1],'green',label = 'Volume')

	leg =plt.legend(fancybox=True, framealpha=0)
	#for text in leg.get_texts():
	#	plt.setp(text, color = 'w')
	plt.xlabel("False positive rate (1- Specificity)")
	plt.ylabel("True positive rate (Sensitivity)")
	plt.title("ROC Curve")
	#plt.title("ROC Curve", color='white')
	plt.savefig(out_path+'ROC_curve.jpeg', bbox_inches='tight',transparent=True)
	plt.show()



#get_roc_for_selected_fetres(df)
	
#get_lasso_and_elasticnet(df)
#do_random_forest(df)
do_TSNE(df,"selected features from PCA")

def random_forest_validation(df):


	valid_df=pd.read_csv('../Data/combat/all/combined_BL/VTASN_testdata.csv')

	train_x, test_x_original, train_y, test_y_original = split_dataset(df)

	header = list(test_x_original)
	test_y = valid_df['group_ID']
	test_x = valid_df.drop(['subj','group_ID'], axis=1)
	#test_x.columns = header

	#roc_value = random_forest_with_selected_features(df,[9],train_x, test_x_original, train_y, test_y)
	#plt.plot(roc_value[0],roc_value[1],'red',label = 'Surface disp.')

	"""new_header = get_new_header(df,[3,6,9])

	train_x = train_x[new_header]
	test_x = test_x[new_header]"""


	# Create random forest classifier instance
	trained_model = random_forest_classifier(train_x, train_y)
	#print("Trained model :: ", trained_model)

	predictions = trained_model.predict(test_x)
	print(test_y)

	for i in range(0, 30):
	    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

	# Train and Test Accuracy
	print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
	print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
	print(" Confusion matrix ", confusion_matrix(test_y, predictions))


	y_predicted = trained_model.predict_proba(test_x)[:,1]
	#test_y1 = list(test_y)
	#dic = {1:0, 2:1}
	#y_test = np.array([dic.get(n, n) for n in test_y1])

	roc_value = roc_curve(test_y, y_predicted)
	plt.plot(roc_value[0],roc_value[1])
	plt.xlabel("False positive rate (1- Specificity)")
	plt.ylabel("True positive rate (Sensitivity)")
	plt.title("ROC Curve")
	plt.savefig(out_path+'ROC_curve.eps', bbox_inches='tight')
	plt.show()

	roc_value = roc_auc_score(test_y, y_predicted)
	print(" ROC area :: ", roc_value)




#random_forest_validation(df)

"""
boston = load_boston()
print(type(boston))
#rf = RandomForestClassifier()
#rf.fit(boston.data[:300], boston.target[:300])
instances = boston.data[[300, 309]]
df = boston.data
print(df)
#df.to_csv("test.csv", sep=',', index=False)

print(np.shape(instances))"""




