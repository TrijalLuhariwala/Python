#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[7]:


df=pd.read_csv("CardioGoodFitness-1.csv")


# In[8]:


df.head()


# In[10]:


df.tail()


# In[13]:


df.sample()


# In[15]:


df.sample(5)


# In[17]:


df.dtypes


# In[19]:


df.info()

df.describe(
# In[24]:


df.describe()


# In[21]:


df.describe(include="all")


# In[26]:


df.isnull().sum()


# # Cardio Good Fitness Case Study

# # business understanding or understanding the stats of business
# ##### Data Understanding
# ##### Data Preparation
# ##### Modeling
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df=pd.read_csv("CardioGoodFitness-1.csv")


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.columns


# In[10]:


df.describe()


# In[9]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df.sample(11)


# In[13]:


sns.boxplot(x="Age",data=df)


# In[15]:


sns.boxplot(x="Education",data=df)


# In[16]:


sns.boxplot(x="Miles",data=df)


# In[17]:


sns.boxplot(x="Usage",data=df)


# In[18]:


sns.boxplot(x="Fitness",data=df)


# In[19]:


sns.boxplot(x="Income",data=df)


# In[20]:


df.describe()


# In[21]:


import warnings
warnings.filterwarnings("ignore")


# In[24]:


sns.displot(df["Age"])  #distribution plot
plt.show()


# In[28]:


df.hist(figsize=(10,20))   ##histogram
plt.show()


# In[ ]:





# In[25]:


df.describe(include="all").T


# In[29]:


sns.countplot(x="MaritalStatus",data=df)


# In[30]:


sns.countplot(x="Product",hue="Gender",data=df)


# In[32]:


sns.countplot(x="Product",data=df)


# In[33]:


sns.countplot(y="Product",data=df)


# In[39]:


sns.countplot(x="Product",hue="MaritalStatus","Gender",data=df)


# In[40]:


sns.boxplot(x="Product",y="Age",data=df)  ##boxplot-one should be numerical


# In[41]:


sns.boxplot(x="Product",y="Income",data=df)


# In[43]:


sns.boxplot(x="Product",y="Miles",data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




