# Feature-Transformation

AIM:

     To read the given data and perform Feature Transformation process and save the data to a file.
     
EXPLANATION:

     Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis. 
     
ALGORITHM:

      Step 1: Read the given data.
      
      Step 2: Clean the Data Set using Data Cleaning Process.
      
      Step 3: Apply Feature Transformation techniques to all the feature of the data set.
      
      Step 4: Save the data to the file.
      
CODE:

Data_to_Transform.csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("/content/Data_to_Transform.csv")

print(df)

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')

plt.show()

sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')

plt.show()

sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')

plt.show()

sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')

plt.show()

df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)

sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df2 = df.copy()

df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df3 = df.copy()

df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df4 = df.copy()

df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)

sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')

plt.show()

from sklearn.preprocessing import PowerTransformer 

trans = PowerTransformer("yeo-johnson")

df5 = df.copy()

df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))

sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution = 'normal')

df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))

sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')

plt.show()

OUTPUT:

![image](https://user-images.githubusercontent.com/123535064/232023655-c4b4c4f2-c409-41f9-99b4-37d1647b48f1.png)

![image](https://user-images.githubusercontent.com/123535064/232023694-425b315d-a1e8-483c-b12e-5e546a2ba41e.png)

![image](https://user-images.githubusercontent.com/123535064/232023721-57c997b7-b45f-4ce3-a547-2961c4e7b688.png)

![image](https://user-images.githubusercontent.com/123535064/232023756-5c489444-6677-45e0-8d25-5daf8373d1fc.png)

![image](https://user-images.githubusercontent.com/123535064/232023798-b7c79019-5fc5-43e3-b598-d8e7c2abc64f.png)

![image](https://user-images.githubusercontent.com/123535064/232023835-d45f9e81-f10c-4312-9e9f-8c2c1611d776.png)

![image](https://user-images.githubusercontent.com/123535064/232023862-350c2863-d538-4da8-a05a-2bb6b0dfe70d.png)

![image](https://user-images.githubusercontent.com/123535064/232023891-9e78f88c-867d-46ed-9c3b-d6cb957cfdfb.png)

![image](https://user-images.githubusercontent.com/123535064/232023916-dbda5732-f595-4092-b98d-3591e7fdb1ef.png)

![image](https://user-images.githubusercontent.com/123535064/232023933-38e02906-3e74-4268-8088-e8f5157498b1.png)

![image](https://user-images.githubusercontent.com/123535064/232023950-5cf0a875-53a1-4944-99bc-6bf59118371a.png)

![image](https://user-images.githubusercontent.com/123535064/232023973-c62c5258-e001-4577-96f4-b89eb73967a3.png)

![image](https://user-images.githubusercontent.com/123535064/232023997-0e319e35-5153-4731-940c-40cacb4f88ae.png)

![image](https://user-images.githubusercontent.com/123535064/232024025-4bf9e507-6f9b-4773-8342-a9a3d4cfa528.png)


RESULT:

      Thus, the Feature Transformation for the given datasets had been executed successfully.

