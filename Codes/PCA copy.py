import numpy as np 
import pandas as pd
import glob
import shutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PD_features import get_side_affected,get_PD_medication,get_UPDRS



#rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 5, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
#plt.rcParams.update(**rc)
#sns.set(rc=rc)



out_path = "../Results/PCA/"

regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]

feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf,disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]
features = [0,3,6,7,8,9]
feature_names = [feature_names[i] for i in features]

feature_len = np.size(features)

#side = False
side = True  #Check the warning later

#standardize = False
standardize = True

num_comp = 5

def get_combined_table(file_list,group): 
	"""Read list of file names in a directory that contains csv files for a group and add them into one.
	file_list: list of paths to csv files
	grou: string specifying the two groups. "CT" or "PD" 

	return: A panda df of combined data"""

	df_base=pd.read_csv(file_list[0])
	subjects = list(df_base['subj'])
	df1 = pd.DataFrame({'subj' : subjects})

	for i in features:

		
		df=pd.read_csv(file_list[i])
		df = df.drop(['age_at_baseline', 'CNO'], axis=1)

		#Looping through all the regions and giving it a new name (append i) at the end to distinguish them
		for cols in regions:
				df.rename(columns={cols: cols+"_"+str(i+1)}, inplace=True)

		df1 = pd.merge(df1, df, on='subj')

	sorted_subj = list(df1['subj'])

	#Giving label names according to the group
	
	if group == "CT":
		group_ID = [1]*np.size(sorted_subj)
	if group == "PD":
		group_ID = [2]*np.size(sorted_subj)
	df1['group_ID'] = group_ID


	#df1.to_csv("../Results/PCA/"+group+"_combined.csv", sep=',', index=False)

	return df1

def get_group_data():

	CT_file_list = glob.glob("../Data/combat/CT/combat_BL"+'/*168*'+'.csv') #finding all the csv files for CT
	CT_file_list.sort()
	for names in CT_file_list:
		print((names.split("/"))[-1] +'\n')

	PD_file_list = glob.glob("../Data/combat/PD/combat_BL"+'/*168*'+'.csv') #finding all the csv files for PD
	PD_file_list.sort()

	#Combining all the tables
	CT_df = get_combined_table(CT_file_list,"CT")
	PD_df = get_combined_table(PD_file_list,"PD")

	return CT_df,PD_df




def get_PCA():
	"""Perform PCA and returns loadings for each feature and explained varience and also plots the
	first to componants. """

	CT_df,PD_df = get_group_data()


	print("Number of CT subjects :", len(CT_df.index))
	print("Number of PD subjects :", len(PD_df.index))


	#This will get labels if the DOMSIDE of PD is needed for PCA
	if side == True:

		PD_Left,PD_Right = get_side_affected(PD_df,"Baseline")

		PD_df = pd.concat([PD_Left,PD_Right],sort=False)
		
		all_df = pd.concat([CT_df, PD_df],sort=False)
		#return_df = pd.concat([CT_df, PD_df],sort=False)
		
		all_df['DOMSIDE'] = all_df['DOMSIDE'].fillna(0)

		#Added these two lines to keep the order of the columns
		sequence = list(PD_df)
		all_df = all_df.reindex(columns=sequence)
		return_df = all_df.reindex(columns=sequence) #This is to return the df before dropping columns
		
		y = list(all_df['DOMSIDE'])
		y = np.array([int(i) for i in y])
		target_names = ["CT","PD_L","PD_R"]
		
		all_df.drop(['subj', 'group_ID','DOMSIDE'], axis=1, inplace=True)
		colors = ['navy', 'turquoise','darkorange']
		n_groups = [0,1,2]


	else:

		all_df = pd.concat([CT_df, PD_df],sort=False)
		return_df = pd.concat([CT_df, PD_df],sort=False) #This is to return the df before dropping columns

		y = np.array(list(all_df["group_ID"]))
		target_names = ["CT","PD"]

		all_df.drop(['subj', 'group_ID'], axis=1, inplace=True)
		colors = ['navy', 'turquoise']

		n_groups = [1,2]

	
	X = np.array(all_df.values)

	if standardize == True:
		X = StandardScaler().fit_transform(X)

	pca = PCA(n_components=num_comp)
	X_r = pca.fit(X).transform(X)

	#Extracting loadingd
	loads = pca.components_


	# Percentage of variance explained for each components
	print('explained variance ratio (first '+ str(num_comp) +' components): %s'
      % str(pca.explained_variance_ratio_))

	return_df.to_csv(out_path+"combined.csv", sep=',', index=False)

	plt.figure()

	lw = 2
	for color, i, target_name in zip(colors, n_groups, target_names):
	    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
	                label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('PCA of PPMI data')

	plt.xlabel('P1')
	plt.ylabel('P2')

	#plt.savefig(out_path+'P1_vs_P2.eps', bbox_inches='tight')
	plt.show()



	return loads,pca.explained_variance_ratio_,return_df


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


def get_plots():
	"""Calling get_PCA and plot figures with loadings"""

	loads,explained_variance,df = get_PCA()


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

	plt.show()



get_plots()

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














