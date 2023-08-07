#!/usr/bin/env python
# coding: utf-8

# # The market research team at AdRight is assigned the task to identify the profile of the typical customer for each treadmill product offered by CardioGood Fitness. The market research team decides to investigate whether there are differences across the product lines with respect to customer characteristics. The team decides to collect data on individuals who purchased a treadmill at a CardioGoodFitness retail store during the prior three months. The data are stored in the CardioGoodFitness.csv file.
# 
# ##The team identifies the following customer variables to study: product purchased, TM195, TM498, or TM798: gender, age, in years;education, in years; relationship status, single or partnered; annual household income; average number of times the customer plans to use the treadmill each week: average number of miles the customer expects to walk/run each week; and self-rated fitness on an 1-to-5 scale, where 1 is poor shape and 5 is excellent shape.

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


## load dataset
df=pd.read_csv("CardioGoodFitness-1.csv")


# In[6]:


df.head()


# In[8]:


df.shape


# # There are 180 rows and 9 columns

# In[9]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df.describe(include="all")


# # top is mode of the data
# #### freq is the freq of the mode
# #### 50% is equivalent to median

# In[14]:


df.hist(figsize=(10,20))
plt.show()


# In[17]:


sns.countplot(x="Product",data=df)


# In[19]:


sns.countplot(x="Product",hue="Gender",data=df)


# In[30]:


sns.countplot(x="Product",hue="MaritalStatus",data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


sns.boxplot(x="Age",data=df)


# In[22]:


sns.boxplot(df)


# In[26]:


sns.boxplot(x="Age",data=df,palette="Set3")


# In[27]:


sns.boxplot(x="Product",y="Age",data=df)


# In[31]:


sns.boxplot(x="Product",y="Income",data=df)


# In[32]:


sns.pairplot(df) ## corelations with each other


# In[33]:


corr=df.corr() ##corelation chart
corr


# In[34]:


sns.heatmap(corr,annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




