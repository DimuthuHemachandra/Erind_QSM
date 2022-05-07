import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters

#df=pd.read_csv('../Results/PCA/combined.csv')

#header = list(df)

#feature_headers = [x for x in header if x not in ['subj', 'group_ID','DOMSIDE','MDS-UPDRS_Total']]


#matrix = np.array(df[feature_headers])

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(50, 75)))

corr = d.corr()

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#plt.imshow(matrix)
plt.show()
