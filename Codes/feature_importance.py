import os

import numpy as np 
import pandas as pd
import scipy.stats  as stats
import glob
import shutil
from PD_features import get_side_affected,get_PD_medication,get_UPDRS,get_any_demo
import seaborn as sns
import matplotlib.pyplot as plt



#rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 5, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
#plt.rcParams.update(**rc)
#sns.set(rc=rc)
sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":18, 'xtick.labelsize': 20,  'ytick.labelsize': 20})  


out_path = "../Results/Training/feature_importance/all/"


def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise

make_out_dir(out_path)

feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf,disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]


#file_list = glob.glob("../Results/Training/feature_importance/"+'*'+'F_P.csv') #finding all the csv files with feature importance

#file_list.sort()


def get_combined_table(file_list,abs=False):
	#This function reads all the csv files and add them into one and return a panda df.
	#Set abs =False to not to get absolute values

	df1=pd.read_csv(file_list[0])
	if abs == True:
		df1['Contribution'] = df1['Contribution'].abs()

	for i in range(1,len(file_list)):

		
		df=pd.read_csv(file_list[i])
		if abs == True:
			df['Contribution'] = df['Contribution'].abs()

		df1 = pd.merge(df1, df, on='Variable')


	return df1


def get_feature_bar_plot(file_list,name):
	#Get the feature importance for all true PD subjects and sort them according to mean value and make a 
	#bar plot of the top 20 features

	added_df = get_combined_table(file_list)


	#Gets mean accross rows
	mean_row = added_df.mean(axis=1)

	#print(mean_row)

	#mean_row = added_df["Contribution"]

	#Inset a new column with the mean.
	added_df.insert(1, "Mean_importance", mean_row, True) 
	added_df = added_df.sort_values(by ='Mean_importance' , ascending=False)

	added_df.to_csv(out_path+"feature_importance.csv", sep=',', index=False)

	new_df = added_df[['Variable','Mean_importance']]
	new_df = new_df.iloc[(-np.abs(new_df['Mean_importance'].values)).argsort()]

	#new_df = pd.concat([new_df.head(10), new_df.tail(10)])
	#new_df = new_df[new_df.Mean_importance.abs()>0.0].sort_values(by=['Mean_importance'])
	new_df = new_df.head(20)
	#new_df = new_df.tail(20)

	plt.figure(figsize=(10, 10))
	sns.barplot(x="Mean_importance", y="Variable", data=new_df)
	plt.savefig(out_path+'mean_importance_'+name+'.jpg', bbox_inches='tight')
	#new_df.plot.barh()
	#plt.show()

names = ['true_P','true_N','F_P','F_N']

#names = ['true_P']

for name in names:

	file_list = glob.glob("../Results/feature_importance/"+'*'+name+'.csv') #finding all the csv files with feature importance

	file_list.sort()

	get_feature_bar_plot(file_list,name)












