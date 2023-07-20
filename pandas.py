#!/usr/bin/env python
# coding: utf-8

# # selection techniques

# In[2]:


import numpy as np


# In[3]:


sample_array=np.arange(1,20)


# In[4]:


sample_array


# In[5]:


sample_array+sample_array


# In[6]:


np.exp(sample_array) #exponential


# In[8]:


np.sqrt(sample_array) #square root


# In[9]:


np.log(sample_array) #logarithm


# In[10]:


np.max(sample_array)


# In[11]:


np.min(sample_array)


# In[12]:


np.argmax(sample_array)


# In[13]:


np.argmin(sample_array)


# In[15]:


np.square(sample_array)


# In[16]:


np.std(sample_array) #standard deviation


# In[17]:


np.var(sample_array) #variance


# In[18]:


np.mean(sample_array)


# In[19]:


array=np.random.rand(3,4)
array


# In[20]:


np.round(array,decimals=2)


# In[21]:


sports=np.array(['golf','cricket','badminton','hockey','cricket'])
np.unique(sports)


# # Pandas

# # helpful to play with arrays
# ###### helpful to arrange the data set

# In[22]:


import pandas as pd
import numpy as np


# In[29]:


sports1=pd.Series([1,2,3,4],index=['cricker','football','baseball','golf'])
sports1


# In[30]:


sports1['football']


# In[ ]:





# In[31]:


sports2=pd.Series([11,2,3,4],index=['cricket','football','baseball','golf'])
sports2


# In[32]:


df1=pd.DataFrame(np.random.rand(8,5),index='A B C D E F G H'.split(),columns='Score1 Score2 Score3 Score4 Score5'.split())
df1


# In[33]:


df1["Score1"]


# In[34]:


sports1+sports2


# In[35]:


df1[["Score1","Score2","Score3"]]


# In[39]:


df1['Score6']=df1['Score1']+df1['Score2']
df1


# In[42]:


df2={'ID':['101','102','103','107','176'],'Name':['John','Kate','akash','kavin','lallu'],'Profit':[20,54,56,87,123]}
df=pd.DataFrame(df2)
df


# In[43]:


df["ID"]


# In[44]:


df[["ID","Profit","Name"]]


# In[46]:


df.drop("ID",axis=1)


# In[47]:


df.drop(3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


df1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




