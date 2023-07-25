#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv("hotel_bookings.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.sample()


# In[7]:


df.sample(6)


# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.describe(include="all")


# In[12]:


df.isnull().sum()


# In[ ]:




