##SECTION 35

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

raw_data=pd.read_csv(r'C:\Users\nickm\Desktop\Regression\1.04. Real-life example.csv')
raw_data.head()

raw_data.describe(include='all')


data=raw_data.drop(['Model'],axis=1)  
data.describe(include='all')

data.isnull().sum()

data_nm=data.dropna(axis=0)
data_nm.describe(include='all')


sns.distplot(data_nm['Price'])

#outliers
q=data_nm['Price'].quantile(0.99)
newdata1=data_nm[data_nm['Price']<op]
newdata1.describe(include='all')
sns.distplot(newdata1['Price'])

sns.distplot(data_nm['Mileage'])
q=newdata1['Mileage'].quantile(0.99)
newdata2=newdata1[newdata1['Mileage']<q]


sns.distplot(data_nm['EngineV'])

newdata3=newdata2[newdata2['EngineV']<6.5]
sns.distplot(newdata3['EngineV'])

q=newdata3['Year'].quantile(0.01)
newdata4=newdata3[newdata3['Year']>q]

data_cleaned=newdata4.reset_index(drop=True)
data_cleaned.describe(include='all')


#OLS Assumptions
f, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()

#Relaxing Assumptions
log_price=np.log(data_cleaned['Price'])
data_cleaned['log_price']=log_price
data_cleaned

f, (ax1,ax2,ax3)=plt.subplots(1,3,sharey=True,figsize=(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')

plt.show()

data_cleaned=data_cleaned.drop(['Price'],axis=1)

#Multicollinearity
data_cleaned.columns.values

variables=data_cleaned[['Mileage','Year','EngineV']]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor(variables.values, i) for i in range (variables.shape[1])]
vif["features"]=variables.columns

data_no_multicollinearity=data_cleaned.drop(['Year'],axis=1)
data_with_dummies=pd.get_dummies(data_no_multicollinearity,drop_first=True)
data_with_dummies.head()

data_with_dummies.columns.values

cols=['Mileage', 'EngineV', 'log_price', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

data_preprocessed=data_with_dummies[cols]
data_preprocessed.head()

#Linear Regression Model

##inputs and targets
targets=data_preprocessed['log_price']
inputs=data_preprocessed.drop(['log_price'],axis=1)

##standardize
scaler=StandardScaler()
scaler.fit(inputs)

inputs_scaled=scaler.transform(inputs)

##Train and test split
x_train, x_test, y_train, y_test=train_test_split(inputs_scaled, targets, test_size=0.2,random_state=365)

##Regression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_hat=reg.predict(x_train)

plt.scatter(y_train,y_hat)
plt.xlabel('Targets(y_train)',size=18)
plt.ylabel('Predictions(y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

sns.distplot(y_train-y_hat)
plt.title("Residuals",size=18)

reg.score(x_train,y_train)
reg.intercept_
reg.coef_

reg_summary=pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights']=reg.coef_
reg_summary

data_cleaned['Brand'].unique()

y_hat_test=reg.predict(x_test)

plt.scatter(y_test,y_hat_test,alpha=0.2)
plt.xlabel('Targets(y_test)',size=18)
plt.ylabel('Predictions(y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

df_pf=pd.DataFrame(y_hat_test,columns=['Prediction'])
df_pf.head()

y_test=y_test.reset_index(drop=True)
y_test.head()

df_pf['Target']=np.exp(y_test)
df_pf

df_pf['Residual']=df_pf['Target']-df_pf['Prediction']
df_pf['Difference%']=np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf.describe()

pd.options.display.max_rows=999
pd.set_option('display.float_format', lambda x:'%.2f' % x)
df_pf.sort_values(by=['Difference%'])