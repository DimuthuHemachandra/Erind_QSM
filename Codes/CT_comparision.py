"""
Purpose : Plot a box_whisker plot comparing healthy controls in PPMI data and VTSAN data (Age and gender matched)

Inputs  : CSV files for both Controls and PD subjects

Outputs : 1) Box_whisker plot and stat results

"""


import numpy as np 
import pandas as pd
from plotting import get_box_plots
from stat_calculator import do_ttest


out_path = "../Results/Control_comparision/"

def get_data(csv_file):
  """Reads a csv file produced by diffparc and returns panda datasets sorted according to healthy
    control list saved in ../Data/VTASN/comparision_subj_list.csv for both PPMI and VTSAN data

  Input : csf_file = string with the file name of a csv file without the extension.

  Output: PPMI dataframe, VTSAN dataframe and string array of region names of striatum"""

  df_comparision = pd.read_csv("../Data/VTASN/comparision_subj_list.csv")
  df_VTASN = pd.read_csv("../Data/VTASN/csv/"+csv_file+".csv")
  df_PPMI = pd.read_csv("../Data/CT/diffparc_2/csv/"+csv_file+".csv")
  concatenated = pd.concat([df_VTASN, df_PPMI])
  df_combined = pd.merge(df_comparision, concatenated, on='subj')

  #Dividing the combined table into two groups
  mask = df_combined['group'] == "PPMI"
  PPMI = df_combined[mask]

  mask = df_combined['group'] == "VTASN"
  VTASN = df_combined[mask]

  #Obtaining region names
  regions=list(PPMI)
  del (regions)[0] #Removing the first 3 elements, because they are not regions
  del (regions)[0]
  del (regions)[0]

  return PPMI,VTASN,regions



#Volume comparision
Analysis = "Volume"
csv_file = "volumes_seed-CIT168striatum_targets-cortical_diffparc_reg-affine_space-MNI152NLin2009cAsym_norm-off"
PPMI,VTASN,regions = get_data(csv_file)

get_box_plots(PPMI,VTASN,regions,Analysis,"Regions",Analysis,out_path,group1_ID="PPMI",group2_ID="VTASN")
do_ttest(PPMI,VTASN,regions,"Baseline",Analysis,out_path)

#Surface displacement comparision
Analysis = "Surface_displacement"
csv_file = "surfdisp_seed-CIT168striatum_targets-cortical"
PPMI,VTASN,regions = get_data(csv_file)

get_box_plots(PPMI,VTASN,regions,Analysis,"Regions",Analysis,out_path,group1_ID="PPMI",group2_ID="VTASN")
do_ttest(PPMI,VTASN,regions,"Baseline",Analysis,out_path)

#FA comparision
Analysis = "FA"
csv_file = "dti-FA_seed-CIT168striatum_targets-cortical_pathways_space-T1w"
PPMI,VTASN,regions = get_data(csv_file)

get_box_plots(PPMI,VTASN,regions,Analysis,"Regions",Analysis,out_path,group1_ID="PPMI",group2_ID="VTASN")
do_ttest(PPMI,VTASN,regions,"Baseline",Analysis,out_path)

#MD comparision
Analysis = "MD"
csv_file = "dti-MD_seed-CIT168striatum_targets-cortical_pathways_space-T1w"
PPMI,VTASN,regions = get_data(csv_file)

get_box_plots(PPMI,VTASN,regions,Analysis,"Regions",Analysis,out_path,group1_ID="PPMI",group2_ID="VTASN")
do_ttest(PPMI,VTASN,regions,"Baseline",Analysis,out_path)






    




