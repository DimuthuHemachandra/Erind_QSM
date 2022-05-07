import numpy as np
import os
import glob
from sort_csv import get_dataframes
from plotting import get_box_plots
from stat_calculator import do_ttest
from PD_features import get_side_affected,get_PD_medication
from UPDRS import get_UPDRS_results
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 15, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
plt.rcParams.update(**rc)
sns.set(rc=rc)


session = "Baseline"
regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]
#combat = True
combat = False

if combat == False:

	CT_data_path = "../Data/CT/diffparc_2/csv/"
	PD_data_path = "../Data/PD/diffparc_2/csv/"
	out_path = "../Results/"
	path_split = "csv/"

else:

	CT_data_path = "../Data/combat/CT/combat_BL/"
	PD_data_path = "../Data/combat/PD/combat_BL/"
	out_path = "../Results/combat/"
	path_split = "combat_BL/"


def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise

#List all the csv file names in PD
PD_csv_file_list = glob.glob(PD_data_path+'*168*'+'.csv') #finding all the csv files
print(PD_csv_file_list)

#sort alphebatically
PD_csv_file_list.sort()

CT_csv_file_list = []
file = []

#Create paths for the CTs with the same file name as PD (This is to avoid having different file names)
for filenames in PD_csv_file_list:

	file_name = filenames.split(path_split)
	#file_name = filenames.split("combat_BL/")
	CT_csv_file_list.append(CT_data_path+file_name[1]) # adding the path to match with CT folder.
	file_type = file_name[1].split(".")
	file.append(file_type[0]) #Getting the just the file name



def get_matrix(session,level="all"):
	P_vals_table = []
	P_sig_table = []
	for i in range(0,np.size(PD_csv_file_list)):

		CT_df,PD_df=get_dataframes(CT_csv_file_list[i],PD_csv_file_list[i],session)
		
		if level == "DOMSIDE_R":
			PD_Left,PD_Right = get_side_affected(PD_df,session)
			PD_df = PD_Right

		if level == "DOMSIDE_L":
			PD_Left,PD_Right = get_side_affected(PD_df,session)
			PD_df = PD_Left

		if level == "PDMED_off":
			PD_med_df,PD_off_med_df,PD_on_med_df = get_PD_medication(PD_df,'PDMEDYN',session)
			PD_df = PD_off_med_df

		if level == "PDMED_on":
			#Does not work for Baseline. Use Month12
			PD_med_df,PD_off_med_df,PD_on_med_df = get_PD_medication(PD_df,'PDMEDYN',session)
			PD_df = PD_on_med_df

		out_path_fig = out_path+file[i]+"/"+level+"/"
		make_out_dir(out_path_fig)
		P_vals,PD_sig = do_ttest(CT_df,PD_df,regions,session,level,out_path_fig,save = "true")
		get_box_plots(CT_df,PD_df,regions,session+'_'+level,"Regions","Volumes",out_path_fig,PD_sig=PD_sig,show = "false")

		P_vals_table.append(P_vals)

		print("PD subjects:",len(PD_df.index))
		
		#Loop through PD_sig to assign values according to significancy
		for i in range(0,np.size(PD_sig)):
			if PD_sig[i] == "-":
				PD_sig[i] = 0
			if PD_sig[i] == "PD_Low":
				PD_sig[i] = -1
			if PD_sig[i] == "PD_High":
				PD_sig[i] = 1
		P_sig_table.append(PD_sig)


	file_no = list(range(1, np.size(file)+1)) #Getting numbers for x axis labels.

	#This section plost P value matric with the P value inside
	##########################################################
	"""
	table = P_vals_table
	df = pd.DataFrame(table)
	df = df.transpose()
	df.columns = file_no
	df = df.assign(regions=regions)
	df = pd.pivot_table(df,index=["regions"])

	f, ax = plt.subplots(figsize=(9, 6))
	sns.heatmap(df, annot=True, linewidths=.5, ax=ax, cmap="vlag")
	#plt.show()
	"""
	##########################################################
	
	#This section plots the significant P val matrix
	##########################################################
	
	table = P_sig_table
	df = pd.DataFrame(table)
	df = df.transpose()
	df.columns = file_no
	df = df.assign(regions=regions)
	df = pd.pivot_table(df,index=["regions"])
	print(df)

	f, ax = plt.subplots(figsize=(9, 6))
	sns.heatmap(df, linewidths=.5, ax=ax, cmap="vlag")
	
	##########################################################

	make_out_dir(out_path+'matrices/')
	plt.savefig(out_path+'matrices/'+level+'.pdf', bbox_inches='tight')
	#plt.subplots_adjust(bottom = 0.6, hspace = 0.05)
	#plt.show()

for names in file:
	print(names +'\n')


#get_matrix(session)

#get_matrix(session,level="DOMSIDE_R")
#get_matrix(session,level="DOMSIDE_L")

#get_matrix(session,level="PDMED_off")
get_matrix("Month12",level="PDMED_on")







