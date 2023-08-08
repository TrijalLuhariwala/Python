#!/usr/bin/env python
# coding: utf-8

# #### Problem statement- to predict the passangers survived or not

# In[220]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[221]:


df=pd.read_csv("titanic-training-data.csv")


# In[222]:


df.shape


# In[223]:


df.head()


# In[224]:


df.info


# In[225]:


df.info()


# In[226]:


df.isnull().sum()


# # Univariate analysis

# ###### univariate

# In[227]:


sns.countplot(x="Survived",data=df,palette="Paired")
plt.show()


# In[228]:


sns.countplot(x="Pclass",data=df,palette="coolwarm")
plt.show()


# In[229]:


sns.countplot(x="Embarked",data=df,palette="coolwarm")
plt.show()


# In[230]:


sns.countplot(x="Sex",data=df,palette="coolwarm")
plt.show()


# In[231]:


import warnings
warnings.filterwarnings("ignore")


# In[232]:


sns.distplot(df["Age"])
plt.show()


# In[233]:


sns.distplot(df["SibSp"])
plt.show()


# In[234]:


sns.distplot(df["Parch"])
plt.show()


# In[235]:


df.describe(include="all")


# ### Bivariate Analysis

# In[236]:


sns.countplot(x="Sex",hue="Pclass",data=df,palette="coolwarm")
plt.show()


# In[237]:


sns.countplot(x="Sex",hue="Embarked",data=df,palette="coolwarm")


# In[238]:


sns.countplot(x="Embarked",hue="Pclass",data=df,palette="coolwarm")


# In[239]:


sns.boxplot(x="Pclass",hue="Age",data=df,palette="coolwarm")


# In[240]:


sns.boxplot(x="Sex",y="Age",data=df,palette="coolwarm")


# In[241]:


sns.boxplot(x="Pclass",y="Age",data=df,palette="coolwarm")


# In[242]:


sns.boxplot(x="Embarked",y="Age",data=df,palette="coolwarm")


# #### Multivariate Analysis

# In[243]:


sns.violinplot(x="Sex",y="Age",hue="Embarked",data=df)


# In[244]:


sns.violinplot(x="Pclass",y="Age",hue="Embarked",data=df,palette="Set3")


# In[245]:


df=df.drop(columns=["PassengerId","Ticket","Fare","Name","Cabin"],axis=1)


# In[246]:


sns.pairplot(df,hue="Survived")


# In[ ]:





# ## Missing value treatement

# In[247]:


median1=df["Age"].median()
df["Age"]=df["Age"].fillna(median1)


# In[248]:


mode1=df["Embarked"].mode()[0]
df["Embarked"]=df["Embarked"].fillna(mode1)


# In[249]:


df.isnull().sum()


# ### Outliers Treatement

# In[250]:


q1=df["Age"].quantile(0.25)
q3=df["Age"].quantile(0.75)
iqr=q3-q1


# In[251]:


lower_limit=q1-1.5*iqr
upper_limit=q3+1.5*iqr


# In[252]:


lower_limit


# In[253]:


df.describe(include="all")


# In[254]:


upper_limit


# In[255]:


df=df[(df["Age"]>lower_limit)&(df["Age"]<upper_limit)]


# In[256]:


sns.boxplot(df["Age"])


# ### Encoding

# In[257]:


df=pd.get_dummies(df,columns=["Sex","Embarked"])


# In[258]:


df.dtypes


# In[259]:


## Horsepower column- ?????? (median)

 ## string cannot be converted to float- astype(float)


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




