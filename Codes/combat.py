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

out_path = "../Results/"
regions = ["Left_Limbic","Right_Limbic","Left_Executive","Right_Executive","Left_Rostral_motor","Right_Rostral_motor","Left_Caudal_motor","Right_Caudal_motor","Left_Parietal","Right_Parietal","Left_Occipital","Right_Occipital","Left_Temporal","Right_Temporal"]

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
PD_csv_file_list = glob.glob("../Data/PD/diffparc_2/csv/"+'*'+'.csv') #finding all the csv files

#sort alphebatically
PD_csv_file_list.sort()

CT_csv_file_list = []
file = []

#Create paths for the CTs with the same file name as PD (This is to avoid having different file names)
for filenames in PD_csv_file_list:

	file_name = filenames.split("csv/")
	CT_csv_file_list.append("../Data/CT/diffparc_2/csv/"+file_name[1]) # adding the path to match with CT folder.
	file_type = file_name[1].split(".")
	file.append(file_type[0]) #Getting the just the file name

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
  print(ID_list)
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


def get_matrix(session):
	P_vals_table = []
	P_sig_table = []
	CT_df_added = pd.DataFrame()
	out_path_CT = "../Data/combat/CT/"+session+"/"
	#make_out_dir(out_path_CT)
	out_path_PD = "../Data/combat/PD/"+session+"/"
	#make_out_dir(out_path_PD)

	#CT_df,PD_df=get_dataframes(CT_csv_file_list[0],PD_csv_file_list[0],session)
	#CT_df= get_center_data(CT_df,session)
	
	
	for i in range(0,np.size(PD_csv_file_list)):


		CT_df,PD_df=get_dataframes(CT_csv_file_list[i],PD_csv_file_list[i],session)
		#Removing subjects who are the only subject from that scanner. (This is because combat.m is crashing if batch = 1)
		one_CT = ["sub-3320_ses-Baseline","sub-4067_ses-Baseline","sub-4085_ses-Baseline"]
		CT_df = CT_df[-CT_df.isin(one_CT)]
		CT_df_c= get_center_data(CT_df,session)
		CT_df_c.drop(['Gender','group'], axis = 1, inplace = True)
		#CT_df = CT_df.set_index('subj').transpose()
	


		#CT_df_c.to_csv(out_path_CT+file[i]+'.csv', sep=',', index=False)


		PD_df_c= get_center_data(PD_df,session)
		PD_df_c.drop(['Gender','group'], axis = 1, inplace = True)
		#PD_df = PD_df.set_index('subj').transpose()



		#PD_df_c.to_csv(out_path_PD+file[i]+'.csv', sep=',', index=False)

		All_df = pd.concat([CT_df, PD_df])
		All_df_c= get_center_data(All_df,session)

		print(list(All_df))

		All_df_c.drop(['Gender'], axis = 1, inplace = True)

		#All_df_c.to_csv('../Data/combat/all/Baseline/'+file[i]+'.csv', sep=',', index=False)

		
			


	return CT_df
	







	#file_no = list(range(1, np.size(file)+1)) #Getting numbers for x axis labels.



df = get_matrix(session)


#df.to_csv('CT_added.csv', sep=',', index=True)









