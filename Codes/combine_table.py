import numpy as np 
import pandas as pd
import scipy.stats  as stats
import glob
import shutil
from PD_features import get_side_affected,get_PD_medication,get_UPDRS,get_any_demo



#rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 5, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
#plt.rcParams.update(**rc)
#sns.set(rc=rc)



out_path = "../Data/combat/all/combined_BL/"

regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]

feature_names = ["FA_norm_max","FA_norm_off","FA_pathways","MD_norm_max","MD_norm_off","MD_pathways","indepConnVolume","surf_area","surf,disp","Vol_MNI_norm_max","Vol_MNI_norm_off","Vol_T1_norm_max","Vol_T1_norm_off"]
features = [2,5,6,7,8,9]
feature_names = [feature_names[i] for i in features]

feature_len = np.size(features)



def get_combined_table(file_list,data='PPMI'): 
	"""Read list of file names in a directory that contains csv files for a group and add them into one.
	file_list: list of paths to csv files

	return: A panda df of combined data"""

	df_base=pd.read_csv(file_list[0])
	subjects = list(df_base['subj'])
	
	if data=='PPMI':
		group = list(df_base['group'])
		df1 = pd.DataFrame({'subj' : subjects, 'group_ID' : group})
	if data=='VTASN':
		group = list(df_base['group_ID'])
		df1 = pd.DataFrame({'subj' : subjects, 'group_ID' : group})

	for i in features:

		
		df=pd.read_csv(file_list[i])
		if data=='PPMI':
			df = df.drop(['age_at_baseline', 'CNO','group'], axis=1)
		if data=='VTASN':
			df = df.drop(['age_at_baseline', 'CNO','group_ID'], axis=1)


		#Looping through all the regions and giving it a new name (append i) at the end to distinguish them
		for cols in regions:
				df.rename(columns={cols: cols+"_"+str(i+1)}, inplace=True)

		df1 = pd.merge(df1, df, on='subj')

	sorted_subj = list(df1['subj'])


	return df1


def get_demographic_stats(CT,PD):

	#CT = CT.dropna()
	#PD = PD.dropna()

	#val = stats.ttest_ind(CT['Age (Years)'], PD['Age (Years)'])

	info_list = ['Age (Years)', 'Years Of Education', 'Duration_Of_Disease(Months)', 'Hoehn-&-Yahr', 'MoCA-Score']

	stat_df = pd.DataFrame(columns=['Group', 'PD', 'Control', 'p-value'])
	stat_df.loc[0] = ['n', PD.shape[0], CT.shape[0], 'nan']

	#CT = CT.dropna()
	#PD = PD.dropna()

	for i,info in enumerate(info_list):

		CT_demo_df = get_any_demo(CT,info,'Baseline')
		PD_demo_df = get_any_demo(PD,info,'Baseline')
		CT_mean = CT_demo_df[info].mean(skipna=True)
		CT_std = CT_demo_df[info].std(skipna=True)
		PD_mean = PD_demo_df[info].mean(skipna=True)
		PD_std = PD_demo_df[info].std(skipna=True)
		val = stats.ttest_ind(CT_demo_df[info], PD_demo_df[info],nan_policy='omit') #val[0] = t value, val[1]= p value
		
		#print(CT_mean,PD_mean,val[1])
		#print(CT_std,PD_std)
		stat_df.loc[i+1] = [info, str(float("%0.2f" % (PD_mean)))+'('+str(float("%0.2f" % (PD_std)))+')', str(float("%0.2f" % (CT_mean)))+'('+str(float("%0.2f" % (CT_std)))+')',float("%0.2f" % (val[1]))]
	#p_val = stats.pearsonr(CT['Age (Years)'], PD['Age (Years)'])
	print(stat_df)
	stat_df.set_index('Group', inplace=True)
	stat_df.to_csv(out_path+"stat.csv", sep=',')
	with open('stat.txt', 'w') as tf:
		tf.write(stat_df.to_latex())



def get_UPDRS_scores(all_df):
	#This function adds UPDRS scores and assign a label to high UPDRS and low UPDRS for training.


	PD = all_df.loc[all_df['group_ID'] == 'PD']
	CT = all_df.loc[all_df['group_ID'] == 'Control']
	#ct['MDS-UPDRS_Total'] = 0 #setting a updrs of 0 for controls
	CT.insert(1, 'MDS-UPDRS_Total', 0)

	CT['group_ID'] = 0
	PD['group_ID'] = 1

	pd_with_updrs = get_UPDRS(PD,"MDS-UPDRS_Total","Baseline")

	get_demographic_stats(CT,pd_with_updrs)
	

	#pd_with_updrs=pd_with_updrs.dropna()
	all_with_UPDRS = pd.concat([CT,pd_with_updrs],sort=False)

	

	updrs = list(all_with_UPDRS['MDS-UPDRS_Total'])

	#Calculating the mean and std for UPDRS
	updrs_filtered = all_with_UPDRS.replace(0, np.nan)
	updrs_filtered = updrs_filtered['MDS-UPDRS_Total'].dropna()
	average_UPDRS = np.divide(list(updrs_filtered),2) # Divide by two to get average UPDRS

	print("mean UPDRS:",np.mean(average_UPDRS)) 
	print("std UPDRS:",np.std(average_UPDRS)) 

	updrs_cutoff = 20

	for i,score in enumerate(updrs):
		if score>0 and score <updrs_cutoff:
			updrs[i] = 1
		if score >= updrs_cutoff or np.isnan(score) :
			updrs[i] = 2 #If the updrs is greater than 20 or nan, asign 2

	all_with_UPDRS.drop('MDS-UPDRS_Total', axis = 1, inplace = True)
	all_with_UPDRS['MDS-UPDRS_Total'] = updrs
	
	early_PD = all_with_UPDRS[all_with_UPDRS['MDS-UPDRS_Total'] != 2]

	return all_with_UPDRS, early_PD



def combine_with_UPDRS():

	all_file_list = glob.glob("../Data/combat/all/combat_BL"+'/*168*'+'.csv') #finding all the csv files for CT
	all_file_list.sort()

	#Printing the file names that have been used to combine
	for names in all_file_list:
		print((names.split("/"))[-1] +'\n')

	all_df = get_combined_table(all_file_list)

	all_with_UPDRS, early_PD = get_UPDRS_scores(all_df)

	#all_with_UPDRS.to_csv(out_path+"Combined_with_UPDRS.csv", sep=',', index=False)

	#early_PD.to_csv(out_path+"Combined_with_earlyPD.csv", sep=',', index=False)


def get_VTASN_combined():
	#Reading harmonized VTASN data and combining them to use as a test dataset.

	vtasn_file_list = glob.glob("../Data/combat/all/testdata_VTASN"+'/*168*'+'.csv') #finding all the csv files for CT
	vtasn_file_list.sort()

	all_df = get_combined_table(vtasn_file_list,data='VTASN') #VTASN tables have both group and group_ID

	all_df.to_csv(out_path+"VTASN_testdata_"+group+".csv", sep=',', index=False)

	ppmi_file_list = glob.glob("../Data/combat/all/train_PPMI_with_VTASN"+'/*168*'+'.csv') #finding all the csv files for CT
	ppmi_file_list.sort()

	print(ppmi_file_list)

	all_df = get_combined_table(ppmi_file_list,data='VTASN') #VTASN tables have both group and group_ID

	all_df.to_csv(out_path+"VTASN_traindata_"+group+".csv", sep=',', index=False)







combine_with_UPDRS()
#get_VTASN_combined()















