from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import pandas as pd

train_data = pd.read_csv('train_ctrUa4K.csv')
# ~ print(train_data.info())

for col in train_data.columns:
    print(col, train_data[col].nunique())

print(train_data.isnull().sum())

train_data['Gender'].fillna(train_data['Gender'].mode().values[0],inplace=True)
train_data['Married'].fillna(train_data['Married'].mode().values[0],inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].mode().values[0],inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode().values[0],inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mean(),inplace=True)
train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(),inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mean(),inplace=True)
# ~ print(train_data.isnull().sum())


test_data=pd.read_csv("test_lAUu6dG.csv")
# ~ print(test_data.isnull().sum())
test_data['Gender'].fillna(test_data['Gender'].mode().values[0],inplace=True)
test_data['Dependents'].fillna(test_data['Gender'].mode().values[0],inplace=True)
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode().values[0],inplace=True)
test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean(),inplace=True)
test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mean(),inplace=True)
test_data['Credit_History'].fillna(test_data['Credit_History'].mean(),inplace=True)
# ~ print(test_data.isnull().sum())


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

train_data['Gender']=encoder.fit_transform(train_data['Gender'])
train_data['Married']=encoder.fit_transform(train_data['Married'])
train_data['Education']=encoder.fit_transform(train_data['Education'])
train_data['Self_Employed']=encoder.fit_transform(train_data['Self_Employed'])
train_data['Property_Area']=encoder.fit_transform(train_data['Property_Area'])
train_data['Loan_Status']=encoder.fit_transform(train_data['Loan_Status'])
# ~ train_data['Loan_ID']=encoder.fit_transform(train_data['Loan_ID'])
train_data['Dependents']=encoder.fit_transform(train_data['Dependents'])
# ~ print(train_data.head())


# ~ test_data['Loan_ID']=encoder.fit_transform(test_data['Loan_ID'])
test_data['Gender']=encoder.fit_transform(test_data['Gender'])
test_data['Married']=encoder.fit_transform(test_data['Married'])
test_data['Education']=encoder.fit_transform(test_data['Education'])
test_data['Self_Employed']=encoder.fit_transform(test_data['Self_Employed'])
test_data['Property_Area']=encoder.fit_transform(test_data['Property_Area'])
test_data['Dependents']=encoder.fit_transform(test_data['Dependents'])
# ~ print(test_data.head())

print(train_data.corr())


x=train_data.drop(columns=['Loan_ID','Loan_Status'])
y_train=train_data['Loan_Status'].values
x_train=x.values

x_test=test_data.drop(columns=['Loan_ID'])
x_test=x_test.values

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=50,random_state=1)

rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
df1=pd.DataFrame(test_data['Loan_ID'],columns=["Loan_ID"])
df1['Loan_Status'] = y_pred
df1=df1.replace(1,'Y')
df1=df1.replace(0,'N')

print(df1)
df1.to_csv("randomforest_output_1.csv",index=False)
