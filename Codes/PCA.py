import numpy as np 
import pandas as pd
import glob
import shutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PD_features import get_side_affected,get_PD_medication,get_UPDRS
from mpl_toolkits.mplot3d import Axes3D




#rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 5, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
#plt.rcParams.update(**rc)
#sns.set(rc=rc)



out_path = "../Results/PCA/"

regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]

feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf,disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]
features = [2,5,6,7,8,9]
#features = [6,7,8]
feature_names = [feature_names[i] for i in features]

feature_len = np.size(features)

UPDRS = False
#UPDRS = True

if UPDRS == False:
    #df=pd.read_csv('../Results/PCA/all_combined.csv')
    df=pd.read_csv('../Data/combat/all/combined_BL/combined_with_UPDRS.csv')
if UPDRS == True:
    df=pd.read_csv('../Results/PCA/combined_with_earlyPD.csv')

standardize = False
#standardize = True

num_comp = 5


def get_new_header(df,feature_list):
	#The reads a list of numbers corresponding to features and concat that number to region names
	#so that it creates a new heasder list to select features from the df.

	header = list(df)

	new_header = []
	for i in feature_list:
		for rois in regions:
			new_header.append(rois+'_'+str(i))

	return new_header


def get_PCA(df):
	"""Perform PCA and returns loadings for each feature and explained varience and also plots the
	first to componants. """

	print("Computing PCA")



	if UPDRS == True:
		y = np.array(list(df["MDS-UPDRS_Total"]))
		df.drop(['subj', 'group_ID','MDS-UPDRS_Total'], axis=1, inplace=True)
		colors = ['navy', 'turquoise','darkorange']
		n_groups = [0,1]
		target_names = ["CT","Early_PD"]
	else:
		y = np.array(list(df["group_ID"]))
		df.drop(['subj', 'group_ID','MDS-UPDRS_Total'], axis=1, inplace=True)
		colors = ['navy', 'turquoise']
		n_groups = [0,1]
		target_names = ["CT","PD"]

	#Uncomment this to get PCA for selected features.
	#new_header = get_new_header(df,[8,9])
	#df = df[new_header]

	X = np.array(df.values)

	if standardize == True:
		X = StandardScaler().fit_transform(X)

	pca = PCA(n_components=num_comp)
	X_r = pca.fit(X).transform(X)

	#Extracting loadingd
	loads = pca.components_


	# Percentage of variance explained for each components
	print('explained variance ratio (first '+ str(num_comp) +' components): %s'
      % str(pca.explained_variance_ratio_))


	fig = plt.figure()


	lw = 2

	########################################
	#for 2D plots
	"""

	for color, i, target_name in zip(colors, n_groups, target_names):
	    plt.scatter(X_r[y == i, 0],X_r[y == i, 1], color=color, alpha=.8, lw=lw,
	                label=target_name)
	#ax.view_init(90, 185)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	#plt.title('PCA of PPMI data')

	plt.xlabel('P1')
	plt.ylabel('P2')
	plt.savefig(out_path+'P1_vs_P2_all.eps', bbox_inches='tight')
	plt.show()"""

	##############################################################
	ax = fig.add_subplot(111, projection='3d')

	for color, i, target_name in zip(colors, n_groups, target_names):
	    ax.scatter(X_r[y == i, 0],X_r[y == i, 1], X_r[y == i, 2], color=color, alpha=.8, lw=lw,
	                label=target_name)
	#ax.view_init(90, 185)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	#plt.title('PCA of PPMI data')

	plt.xlabel('P1')
	plt.ylabel('P2')
	ax.set_zlabel('P3')

	plt.savefig(out_path+'PCA_all.eps', bbox_inches='tight')
	#plt.show()

	# rotate the axes and update
	for angle in range(0, 360):
	    ax.view_init(30, angle)
	    plt.show()
	    plt.pause(.001)

	#######################################################

	return loads,pca.explained_variance_ratio_


def get_feature_loadig_plots(component,name):
	"""Get loadings sorted according to features and plot them with different colors
		component: list of loadings for each component
		name: (string) Name of the component"""

	j=0
	for i in range(0,feature_len):
		feat_i = component[j:j+np.size(regions)]
		l = list(range(j,j+np.size(regions)))
		plt.bar(l,feat_i)
		plt.ylabel(name)
		j = j+np.size(regions)

def get_region_loading_plots(component,name):
	"""Get loadings sorted according to regions and plot them with different colors
		component: list of loadings for each component
		name: (string) Name of the component"""


	k = 0
	for i in range(0,np.size(regions)):
		region_i = component[i::14]
		l = list(range(k,k+feature_len))
		plt.bar(l,region_i, label = regions[i])
		plt.ylabel(name)
		k = k+feature_len


def get_plots(df):
	"""Calling get_PCA and plot figures with loadings"""

	loads,explained_variance = get_PCA(df)


	#Plotting loadings sorted by features
	for i in range(0,num_comp):

		ax = plt.subplot(num_comp,1,i+1)
		if i == 0:
			get_feature_loadig_plots(loads[i,:],"PC"+str(i+1))
			ax.legend(feature_names,loc='lower left',bbox_to_anchor=(0,1.02,1,0.2),
			          fancybox=True, ncol=14, prop={'size': 6})
		else:
			get_feature_loadig_plots(loads[i,:],"PC"+str(i+1))

	#plt.savefig(out_path+'loadings_per_feature.eps', bbox_inches='tight')
	#plt.show()

	#Plotting loadings sorted by regions
	"""
	for i in range(0,num_comp):

		ax = plt.subplot(num_comp,1,i+1,sharex = ax)
		if i == 0:
			get_region_loading_plots(loads[i,:],"PC"+str(i+1))
			ax.legend(regions,loc='lower left',bbox_to_anchor=(0,1.02,1,0.2),
			          fancybox=True, ncol=14, prop={'size': 6})
		else:
			get_region_loading_plots(loads[i,:],"PC"+str(i+1))

	plt.subplots_adjust(wspace=0.01)
	plt.savefig(out_path+'loadings_per_region.eps', bbox_inches='tight')

	plt.show()"""



get_plots(df)

def select_features():

	loads,explained_variance,df = get_PCA()
	loads = np.absolute(loads)
	max = np.max(loads)
	selected = np.where(loads >max*0.9)[1] #Getting y coordinates of the loadings sorted with greater than 80% of the max loading
	selected = list(set(selected)) #Removing duplicates
	selected.sort()
	if side == True:
		info = df[['subj','DOMSIDE','group_ID']]
	else:
		info = df[['subj','group_ID']]

	df = df.iloc[:, selected]
	df = pd.concat([info,df],sort=False, axis=1)
	df.to_csv(out_path+"selected_features.csv", sep=',', index=False)



#select_features()

def get_UPDRS_scores():
	#This function adds UPDRS scores and assign a label to high UPDRS and low UPDRS for training.

	CT_df,PD_df = get_group_data()

	PD_df = get_UPDRS(PD_df,"MDS-UPDRS_Total","Baseline")
	PD_df=PD_df.dropna()

	all_df = pd.concat([CT_df, PD_df],sort=False)
		
	all_df['MDS-UPDRS_Total'] = all_df['MDS-UPDRS_Total'].fillna(0)

	updrs = list(all_df['MDS-UPDRS_Total'])

	for i,score in enumerate(updrs):
		if score>0 and score <20:
			updrs[i] = 1
		if score >= 20:
			updrs[i] = 2

	all_df.drop('MDS-UPDRS_Total', axis = 1, inplace = True)
	all_df['MDS-UPDRS_Total'] = updrs

	all_df.to_csv(out_path+"Combined_with_UPDRS.csv", sep=',', index=False)


	updrs_df = all_df[['subj','MDS-UPDRS_Total']]

	return updrs_df



#get_UPDRS_scores()

"""
all_file_list = glob.glob("../Data/combat/all/combat_BL"+'/*168*'+'.csv') #finding all the csv files for CT
all_file_list.sort()

all_df = get_combined_table(all_file_list,"all")

updrs = get_UPDRS_scores()

df1 = pd.merge(all_df, updrs, on='subj')

df1.to_csv(out_path+"Combined_with_UPDRS_all.csv", sep=',', index=False)
"""

#loads,explained_variance,df = get_PCA()	














