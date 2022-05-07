"""
Purpose : Plot a box_whisker plot for two groups

Inputs  : CSV files for two groups

Outputs : 1) Box_whisker plot for diffparc values of different groups



"""
import numpy as np 
from pylab import *              
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.interactive(False)



def get_significant_regions(PD_sig,regions):

  sig_regions = []
  for i,p in enumerate(PD_sig):
    
    if p!= "-":
      sig_regions.append(i+i+1) #index goes from 1 to 14, but this adjsut the index to get PD only
    

  return sig_regions




def get_box_plots(group1,group2,regions,plot_name,x_title,y_title,out_path,group1_ID="CT",group2_ID="PD",PD_sig=0,show = "true"):
  """Plots boxplots for two groups for given regions.
  regions: List of region names (strings) match with the header in CSV file"""
  
  rc={'axes.labelsize': 15, 'font.size': 15, 'legend.fontsize': 5, 'axes.titlesize': 15 , 'xtick.labelsize': 15, 'ytick.labelsize': 15}
  plt.rcParams.update(**rc)
  sns.set(rc=rc)

  sig_regions = get_significant_regions(PD_sig,regions)
  print(sig_regions)

  concatenated = pd.concat([group1.assign(dataset=group1_ID), group2.assign(dataset=group2_ID)]) #Adding two tables together
  #print(concatenated)
  Melted_data = concatenated.melt(id_vars=['subj','dataset'], value_vars=regions,var_name='regions', value_name='volumes') #melting fields


  fig, ax = plt.subplots()
  p = sns.boxplot(x='regions',y='volumes',data=Melted_data, hue='dataset', palette= "muted")
  for index in sig_regions:  
    mybox = p.artists[index]

    colors=["#F00000","#00A000"] #Red and Green
    color_dict = dict(zip(colors, colors)) #Making a colur pallete 
    mybox.set_facecolor(color_dict[colors[0]])
  #plt.plot(sig_regions,[1000 for i in sig_regions],"o")

  #p = sns.stripplot(data=PD, jitter=True);
  #p = sns.stripplot(data=CT, color="b", jitter=True);

  plt.xticks(rotation=70)
  plt.savefig(out_path+plot_name+'.pdf', bbox_inches='tight')
  plt.subplots_adjust(bottom = 0.3)
  ylabel(y_title, fontsize=16)
  xlabel(x_title, fontsize=16)
  plt.legend(loc='upper center', bbox_to_anchor=(0.9,0.9),prop={'size':15})
  if show == "true":
    plt.show()
 






  


























    
        
      
    




