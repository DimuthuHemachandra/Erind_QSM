"""
Purpose : To read a dataframe corresponding to Diffparc_sumo results and analyze them with UPDRS and other scores

Inputs  : Datframe for PD subjects

Outputs : 1) Line plot for diffpac resultrs vs each score 
          2) Stats of each plot in a txt file
 

Notes  : 

"""
import numpy as np 
import scipy                  
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import os
from PD_features import get_UPDRS
from sort_csv import get_dataframes


rc={'axes.labelsize': 5, 'font.size': 5, 'legend.fontsize': 1, 'axes.titlesize': 5 , 'xtick.labelsize': 1, 'ytick.labelsize': 1}
plt.rcParams.update(**rc)
plt.rcParams.update({'figure.max_open_warning': 0})
sns.set(rc=rc)


def get_data(PD_df,UPDRS_Nu,session):
  """TReads a data frame with PD data and returns another data frame with a UPDRS(or other) score"""

  df = get_UPDRS(PD_df,UPDRS_Nu,session)
  df=df.dropna()

  return df



def get_fitted_plots(df,x_UPDRS,region_names,session,Analysis,out_path):
  """Returns a line plot and stat of the fitted line
  df: dataframe
  x_UPDRS: String,column name for UPDRS (or other) scores.
  region_names: String array with striatal region names
  session: String (eg: "Baseline")
  Analysis: String (Name for the analysis)
  out_path: path to the output folder
  """

  for regions in region_names:

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.lmplot(x=x_UPDRS, y=regions,data=df,fit_reg=True) 

    X = list(df[x_UPDRS])
    Y = list(df[regions])
    upY= np.max(Y)
    lowY = np.min(Y)
    plt.ylim(lowY,upY)
    results = sm.OLS(Y, X).fit()
    stat_results=results.summary()
    R_val=round((results.rsquared),2)
    plt.title(R_val,loc='center')
    with open(out_path+session+"_"+Analysis+"_"+regions+"_vs_"+x_UPDRS+"_plot_stats.txt", "w") as text_file:
      print(stat_results, file=text_file)
    plt.savefig(out_path+session+"_"+Analysis+"_"+regions+"_vs_"+x_UPDRS+'_plot.pdf')
    #plt.show()



def get_UPDRS_results(PD_df,regions,session,Analysis,out_path):
  """Returns a line plot and stat of the fitted line
  PD_df: dataframe for PD data
  region_names: String array with striatal region names
  session: String (eg: "Baseline")
  Analysis: String (Name for the analysis)
  out_path: path to the output folder
  """

  x_parameters = ["MDS-UPDRS_Total","MDS-UPDRS_Part-I","MDS-UPDRS_Part-I-Patient-Questionnaire","MDS-UPDRS_Part-II-Patient-Questionnaire","MDS-UPDRS-Part-III-Patient-Questionnaire","Hoehn-&-Yahr","Modified-Schwab-&-England-ADL","UPSIT-Total-Score","MoCA-Score","GDS-Score","SCOPA-AUT","SBR-Left-Caudate","SBR-Right-Caudate","SBR-Left-Putamen","SBR-Right-Putamen"]

  for x_vals in x_parameters:
    filename = out_path+Analysis+"/"+x_vals+"/"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
              raise

    df = get_data(PD_df,x_vals,session)
    get_fitted_plots(df,x_vals,regions,session,Analysis,filename)


#########################################################################################









    
        
      
    




