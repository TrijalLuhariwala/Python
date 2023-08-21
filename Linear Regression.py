#!/usr/bin/env python
# coding: utf-8

# # Problem Statement- we will conduct a linear model that explains the relationship between a cars miles with its other attributes

# #### Step one- import libraries

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ### Step two- load file

# In[5]:


df=pd.read_csv("auto-mpg.csv")


# In[6]:


df.shape


# In[7]:


df.sample(10)


# In[8]:


df.drop("car name",axis=1,inplace=True)


# #### Also replace the categorical var with actual values

# In[9]:


df['origin']=df['origin'].replace({1:'america',2:'europe',3:'asia'})


# In[10]:


df.sample(10)


# In[11]:


df.isnull().sum()


# In[12]:


df.dtypes


# ### Deal with missing values

# In[13]:


### One hot encoding
df=pd.get_dummies(df,columns=['origin'])


# In[14]:


df.dtypes


# In[15]:


df.describe(include="all")


# In[16]:


df.describe()


# #### hp(horsepower) missing because it does not seem to be recognized as a numerical column24

# In[17]:


##isdigit()? on horsepower
hpIsDigit=pd.DataFrame(df.horsepower.str.isdigit())   ##if stings is made of digits store True else False

#print isDigit= False
df[hpIsDigit['horsepower']==False]   ##from temp take only those rows where hp has false


# In[18]:


df["horsepower"]=df["horsepower"].replace("?",np.nan)
df["horsepower"]=df["horsepower"].astype(float)


# In[19]:


median1=df["horsepower"].median()
median1


# In[20]:


df["horsepower"].replace(np.nan,median1,inplace=True)


# In[21]:


df[hpIsDigit['horsepower']==False]


# In[22]:


df.dtypes


# In[23]:


#### DUplicates
duplicate=df.duplicated()
duplicate.sum()


# In[24]:


## there are various ways to treat missing values such as replacing the missing values with mean median etc.


# ## Bivariate Plotes

# ##### A bivariate analysis can be done using scatter mix plot.Seaborn libs create a dashboard reflecting useful in the dimensions.the result can be stored as png files

# In[25]:


sns.pairplot(df,diag_kind='kde')


# #### Observation between 'mpg' and other attributes indicate the relationship is not really linear. however the plots also indicate that liniarity is quite a bit of useful information/pattern.Several assumptions of classical linear regression seem to be violated,including the assumptions and heteroscedasticity

# ## Split data

# In[26]:


## Lets build linear model
# independent variables
X=df.drop(['mpg'],axis=1)
#dependent variable
y=df[['mpg']]


# In[28]:


## Split X and y into train and test in 70:30 ratio

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)


# ### Fit linear model

# In[29]:


model_1=LinearRegression()
model_1.fit(X_train,y_train)


# ##### Here are coefficients for each variable and the intercept

# ##### The score R^2 for in in-sample and out of the sample

# In[30]:


model_1.score(X_train,y_train)


# In[31]:


## out of sample score (R^2)

model_1.score(X_test,y_test)


# In[33]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


# In[38]:


poly=PolynomialFeatures(degree=2,interaction_only=True)
X_train2=poly.fit_transform(X_train)
X_test2=poly.fit_transform(X_test)

poly_clf=linear_model.LinearRegression()

poly_clf.fit(X_train2,y_train)

#y_pred=poly_clf.predict(X_test2)
#print(y_pred)
#In sample training R^2 will alway improve with the number of variables

print(poly_clf.score(X_train2,y_train))


# In[42]:


## Out of sample testing R^2 is our measure of success and does improve
print(poly_clf.fit(X_test2,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




