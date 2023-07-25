#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("auto-mpg.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.sample()


# In[6]:


df.sample(7)


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.describe(include="all")


# In[12]:


df.isnull().sum()


# In[ ]:





# In[ ]:




