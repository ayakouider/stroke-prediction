# stroke-prediction
To get started load these libraries:
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
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
## visualization:
this line is to transform the type to category so it doesnt cause an error in graphing
```
data['gender'] = data['gender'].astype('category')
```
for categorical variables we used countplot:
```
categorical_col= ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_col:
    ax = sns.countplot(x = col, data = data)
    total = len(data)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.4f}%\n'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center')
    plt.tight_layout()
    plt.show()
    print("*" * 50)
```
and for numerical variables we used histograms and kdeplot:
```
numerical_col= ['avg_glucose_level', 'bmi', 'age']
for col in numerical_col:
    sns.histplot(x = col, data = data, kde = True)
    plt.show()
    print("*" * 50)
    
for col in numerical_col:
    sns.kdeplot(x = col, data = data, hue='stroke')
    plt.show()
    print("*" * 50)
```
## Training:
first we split the data and scale it:
```
y=data["stroke"]
x=data[["hypertension","bmi","heart_disease","gender_Female","gender_Male","gender_Other","ever_married","avg_glucose_level","age","smoking_status_Unknown","smoking_status_formerly smoked","smoking_status_never smoked","smoking_status_smokes"]]

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

```
the model we used is gradient boosting:
```
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(learning_rate=0.05,max_depth=2,min_samples_split=0.3,n_estimators=100,max_features=12)
gb.fit(Xtrain,ytrain)
```
the performance metric we used is the auroc curve:
```
pred3=gb.predict_proba(Xtest)
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc=roc_auc_score(ytest,pred3[:,1])
fpr,tpr,_=roc_curve(ytest,pred3[:,1])
plt.plot(fpr,tpr,marker=".",label='AUROC=%0.3f'%auc)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
```
## testing:
we repeat the steps of loading and prepairing the test data.
then we scale it so we can predict the likelihood of a stroke.
scaling:
```
xts=sc.transform(xt)
```
prediction:
```
result=gb.predict(xts)
```
confidence interval:
```
st.t.interval(0.90, len(result)-1, loc=np.mean(result), scale=st.sem(result))
st.t.interval(0.95, len(result)-1, loc=np.mean(result), scale=st.sem(result))
```
then we put the results in a csv file with the id:
```
idr=testD.index
with open('project.csv','w+') as file:
    myfile=csv.writer(file)
    myfile.writerow(['id','stroke'])
    for i in range(len(idr)):
        myfile.writerow([idr[i],result[i]])
```

