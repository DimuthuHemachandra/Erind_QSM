import numpy as np 
import pandas as pd
from scipy import stats


def outlier_detect(df):
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(df[i])
        p=[]
        for j in x:
            if j < LTV or j>UTV:
                p.append(df[i].median())
            else:
                p.append(j)
        df[i]=p
    return df

def split(df, headSize) :
    CT = df.head(headSize)
    PD = df.tail(len(df)-headSize)
    CT = outlier_detect(CT) # without outliers
    PD = outlier_detect(PD) # without outliers
    df = pd.concat([CT, PD],sort=False)
    return df

def remove_outliers(df):

	subj = df['subj']
	df = df.drop('subj' , axis='columns')

	df = split(df,21)
	df = df.join(subj, how='outer')
	cols = list(df.columns)
	cols = [cols[-1]] + cols[:-1]
	df = df[cols]
	df = df.dropna()

	return df

