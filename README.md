# stroke-prediction
To get started load these libraries:
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
```
## loading data:
```
df1=pd.read_csv(r"C:\Users\tuma2\Downloads\healthcare-dataset-stroke-data.csv",index_col="id")
df2=pd.read_csv(r"C:\Users\tuma2\Downloads\train.csv",index_col="id")
data=pd.concat([df1,df2],axis=0)
```
concat function is used to merge the 2 dataframes vertically.
## preparing the data:
to deal with null values in bmi we used interpolate() function which replaces the null values with the mean
```
data["bmi"].interpolate(method="linear",inplace=True)
```
we droped work_type and residence_type
```
data=data.drop(columns=["Residence_type","work_type"])
```
we one-hot-encoded nominal data using pandas get_dummies()
```
data=pd.get_dummies(data,columns=["gender","smoking_status"])
```
for ordinal data we used a dictionary to encode the data and replace it in the dataframe
```
int_marr={"ever_married":{"Yes":1,"No":0}}
data.replace(int_marr,inplace=True)
```
using seaborns countplot() for categorical variables

and kdeplot() for numerical variables

4.Scaling:

split the data using train_test_split()

scale the data by fiting it to StandardScaler()

5.Model:
 fit the data to  GradientBoostingClassifier()

6. performance metric:
 use predict the result of the ytest using predict_proba()
 
 use auc_roc_score(),roc_curve() and auc to get the score 
 
7. test:
repeat the data cleaning and transform the test data so you can get the final prediction

export it as a csv file

