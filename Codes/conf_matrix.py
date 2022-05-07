import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

out_path = '../Results/Training/'
def get_matrix(array,title):
    df_cm = pd.DataFrame(array, index = [i for i in ["CT","PD"]],
                      columns = [i for i in ["CT","PD"]])
    plt.figure(figsize = (4,4))
    ax = sn.heatmap(df_cm, annot=True, cbar=False, cmap="YlGnBu", annot_kws={"size": 20})

    #titles and legends
    ax.set_title('Confusion matrix of '+title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    plt.tight_layout()  #set layout slim
    plt.savefig(out_path+'confusion_matrix_'+title+'.pdf', bbox_inches='tight')
    plt.show()

def get_vals(df):
    
    print(data.values)
    array = np.zeros((2, 2))
    array[0,0] = data.values[0]
    array[1,0] = data.values[1]
    array[0,1] = data.values[2]
    array[1,1] = data.values[3]

    return array

    
title = "PD vs controls (PPMI)"
data = pd.read_csv(out_path+'confusion_matrix.csv', header =None)
array = get_vals(data)
get_matrix(array,title)


title = "early PD vs controls (PPMI)"
data = pd.read_csv(out_path+'confusion_matrix_early_PD.csv', header =None)
array = get_vals(data)
get_matrix(array,title)

title = "PD vs controls (UWO)"
data = pd.read_csv(out_path+'confusion_matrix_VTASN.csv', header =None)
array = get_vals(data)
get_matrix(np.rint(array/5),title)


