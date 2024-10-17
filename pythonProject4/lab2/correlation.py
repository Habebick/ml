import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

brainFrame = pd.read_csv('brainsize.txt', delimiter='\t')
print(brainFrame)
menDf = brainFrame[brainFrame['Gender'] == 'Male']
womenDf = brainFrame[brainFrame['Gender'] == 'Female']

#womenMeanSmarts = womenDf[["PIQ", "FSIQ", "VIQ"]].mean(axis=1)
#plt.scatter(womenMeanSmarts, womenDf["MRI_Count"])
#plt.show()
#menMeanSmarts = menDf[["PIQ", "FSIQ", "VIQ"]].mean(axis=1)
#plt.scatter(menMeanSmarts, menDf["MRI_Count"])
#plt.show()
#print(womenDf)
#print(menDf[['PIQ','FSIQ','VIQ','Weight','Height','MRI_Count']].corr(method='pearson'))
mcorr = menDf[['PIQ','FSIQ','VIQ','Weight','Height','MRI_Count']].corr()
sns.heatmap(mcorr)
plt.show()