# stroke-prediction
1.loading data:

using pandas read_csv() function to load both datasets

concat() function to join data vertically

2.data visualization:

using seaborns countplot() for categorical variables

and kdeplot() for numerical variables

3.data cleaning:

to deal with null values in bmi we used interpolate() function which replaces the null values wit the mean

we droped work_type and residence_type

we one-hot-encoded gender and smoking status 

labeled ever_married using a dictionary to achieve the same results as ordinal encoding

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

