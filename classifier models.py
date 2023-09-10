#!/usr/bin/env python
# coding: utf-8

# ## Problem statement: to predict wheather the loan is approved or not

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("loan_prediction.csv")


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


sns.countplot(x="Gender",hue="Loan_Status",data=df)


# In[6]:


sns.countplot(x="Married",hue="Loan_Status",data=df)


# In[7]:


df.drop("Loan_ID",axis=1)


# In[8]:


df.describe()


# In[9]:


#Filling all Nan values with mode of respective variable

df["Gender"].fillna(df[ "Gender"].mode()[0], inplace=True)
df["Married"]. fillna (df["Married"]. mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)
df["Dependents"].fillna (df[ "Dependents"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

#ALL values of "Dependents" columns were of "str" form now converting to "int" form. data["Dependents"] = data["Dependents"].replace('3+, int(3))
df["Dependents"] = df["Dependents"].replace('3+', int(3))
df["Dependents"] = df["Dependents"].replace('1', int(1))
df["Dependents"] = df[ "Dependents"].replace('2', int (2))
df["Dependents"] = df["Dependents"].replace('0', int(0))
df["LoanAmount"].fillna (df["LoanAmount"].median (), inplace=True)

print(df.isnull().sum())

#Heat map for null values

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull())


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[11]:


df["Gender"] = le.fit_transform(df[ "Gender"])
df["Married"]=le.fit_transform(df["Married"])
df["Education"]=le.fit_transform(df["Education"])
df["Self_Employed"]= le.fit_transform(df["Self_Employed"])
df["Property_Area"]=le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])


# In[12]:


X=df.drop(["Loan_Status","Loan_ID"],axis=1)
y=df["Loan_Status"]


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# model1=LogisticRegression(solver="libilinear")

# In[15]:


model1=LogisticRegression(solver="liblinear")


# In[16]:


model1.fit(X_train,y_train)


# In[17]:


model1.score(X_train,y_train)


# In[18]:


model1.score(X_test,y_test)


# ### Decision tree

# In[19]:


from sklearn.tree import DecisionTreeClassifier


# In[20]:


model2=DecisionTreeClassifier(max_depth=3)  ## Adjust depth acc to the score


# In[21]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# In[22]:


model2.fit(X_train,y_train)


# In[23]:


model2.score(X_train,y_train)


# In[24]:


model2.score(X_test,y_test)


# ### Confusion matrix

# In[25]:


from sklearn import metrics


# In[26]:


dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[27]:


dtree.score(X_train,y_train)


# In[28]:


dtree.score(X_test,y_test)


# In[29]:


y_predict=dtree.predict(X_test)


# In[30]:


cm=metrics.confusion_matrix(y_test,y_predict,labels=[0,1])

df_cm=pd.DataFrame(cm,index=[i for i in["No","Yes"]],
                  columns=[i for i in["No","Yes"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# ## Bagging model

# In[33]:


from sklearn.ensemble import BaggingClassifier
bgcl=BaggingClassifier(n_estimators=250)  ### n_estimators is the no of models you want to run the classifier
bgcl= bgcl.fit(X_train, y_train)
Ty_predict=bgcl.predict(X_test)
print(bgcl.score(X_test,y_test))


# In[34]:


print(bgcl.score(X_train,y_train))


# ## Ada boost classifier and gradient boosting classsifier

# In[50]:


from sklearn.ensemble import AdaBoostClassifier
abcl=AdaBoostClassifier(n_estimators=25)  ### n_estimators is the no of models you want to run the classifier
abcl= abcl.fit(X_train, y_train)
Ty_predict=abcl.predict(X_test)
print(abcl.score(X_test,y_test))


# In[51]:


print(abcl.score(X_train,y_train))


# In[52]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl=GradientBoostingClassifier(n_estimators=250)  ### n_estimators is the no of models you want to run the classifier
gbcl= gbcl.fit(X_train, y_train)
Ty_predict=gbcl.predict(X_test)
print(gbcl.score(X_test,y_test))


# In[53]:


gbcl.score(X_train,y_train)


# ### Random Forest Model

# In[62]:


from sklearn.ensemble import RandomForestClassifier
rfcl=RandomForestClassifier(n_estimators=150,max_features=6)
rfcl.fit(X_train,y_train)


# In[63]:


rfcl.score(X_train,y_train)


# In[64]:


rfcl.score(X_test,y_test)


# In[66]:


from sklearn.ensemble import RandomForestRegressor
rfrs=RandomForestRegressor(n_estimators=100,max_features=6)
rfrs.fit(X_train,y_train)


# In[67]:


rfrs.score(X_train,y_train)


# In[68]:


rfrs.score(X_test,y_test)


# In[ ]:




