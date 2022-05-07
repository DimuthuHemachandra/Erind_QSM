"""
Purpose : Plot a box_whisker plot for CT and PD data

Inputs  : CSV files for both Controls and PD subjects

Outputs : 1) Box_whisker plot for diffparc values of PD and CT



"""
import numpy as np   
import scipy                
import scipy.stats as stats
import statsmodels.stats.multitest as smm
import statsmodels.formula.api as sm
import pandas as pd
#import statmodels.api as sm
#import statmodels.formula.api as ols



def reject_outliers(data, m = 3.):
    """Eliminate outliers that are not within 2 sigma and returns.
    Data: 1D array of data
    m: specify the sigma"""
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def get_stat(CT,PD,volume_names):
  """Performs t test and between CT and PD subjects and returns t values and p values.
  CT: datastructre for control subjects
  PD: datastructure for PD subjects
  regions: List of region names (strings) match with the header in CSV file"""

  CTval = CT[volume_names].tolist()
  PDval = PD[volume_names].tolist()

  #reject_outliers(np.array(CTvol))
  CTval=reject_outliers(np.array(CTval)) #Removing outliers. Reject_outlier methods takes only arrays
  PDval=reject_outliers(np.array(PDval))

  vals = scipy.stats.ttest_ind(PDval,CTval)

  #Returning the t-value and the P-values
  return vals[0], vals[1]


def do_ttest(CT_vals,PD_vals,regions,session,filename,out_path,save = "true"):
  #Statistics section
  P_val_list = []
  t_val_list = []
  PD_sig = []

  #Calling get_stat for each regions and saves p values into a txt file
  for j,volume_names in enumerate(regions):

    t_val, P_val = get_stat(CT_vals,PD_vals,volume_names)
    P_val_list.append(P_val)

    if t_val >= 0.0 and P_val <=0.055:
      PD_sig.append("PD_High")
    if t_val < 0.0 and P_val <=0.055:
      PD_sig.append("PD_Low")
    if P_val >0.055:
      PD_sig.append("-")


  rej, pval_corr,a,b = smm.multipletests(P_val_list, alpha=0.05, method='bonferroni')

  if save == "true":
    with open(out_path+session+'_'+filename+'_stat_results.txt', 'w') as f:
        for f1, f2, f3 in zip(regions,P_val_list,pval_corr):
            print(f1,"\t" ,f2,"\t",f3, file=f )
    stat_df = {'Region':regions,'P Value':P_val_list,'Corrected P Value':pval_corr,'Significancy':PD_sig}
    stat_df = pd.DataFrame(stat_df)
    #stat_df.reset_index(drop=True, inplace=True)
    stat_df.set_index('Region', inplace=True)
    print(stat_df)
    with open(out_path+session+'_'+filename+'_stat_results_latex.txt', 'w') as tf:
        tf.write(stat_df.to_latex())

  return P_val_list,PD_sig





























    
        
      
    




