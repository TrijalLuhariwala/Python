#!/usr/bin/env python
# coding: utf-8

# # Numpy, Package for multidimensional array

# # NUM stands for numerical PY stands for python

# # helps to play with numbers

# # helps in creating arrays

# In[4]:


import numpy as np


# In[6]:


sample_list=[6,7,9]
np.array(sample_list)


# In[7]:


arr=np.array([1,2,3])
arr


# In[8]:


list_of_lists=[[1,2,3],[4,5,6],[7,8,9]]
np.array(list_of_lists)


# In[10]:


np.arange(5,10)


# In[11]:


np.arange(5,3)


# In[12]:


np.arange(1,100)


# In[14]:


np.arange(1,31,5)


# In[15]:


np.arange(5)


# In[16]:


np.zeros(10)


# In[17]:


np.ones((3,4))


# In[19]:


np.ones(10,int)


# In[20]:


np.zeros((2,6),int)


# In[25]:


np.ones((2,5))


# # linspace, linearly spaced

# In[26]:


np.linspace(0,5)


# In[27]:


np.linspace(0,2,5)


# # np.linspace(starting no,last no,no of outputs)

# In[28]:


np.linspace(0,20,8)


# In[30]:


np.eye((11))


# In[31]:


np.random.rand(3,5)


# In[32]:


arr=np.random.rand(2,4)
arr


# In[34]:


np.random.randint(2,100)


# In[35]:


np.random.randint(20,55,100)


# In[38]:


sample_array=np.arange(30)
sample_array


# In[40]:


rand_array=np.random.randint(0,100,20)
rand_array


# In[43]:


sample_array.reshape(5,6)


# In[44]:


sample_array.rshape(9,2)


# In[46]:


rand_array.argmax()


# In[47]:


rand_array.max()


# In[48]:


a=np.eye(5)
a


# In[50]:


a.T


# In[52]:


a=np.random.rand(2,3)
a


# In[53]:


a.T


# In[54]:


sample_array=np.arange(10,21)
sample_array


# In[56]:


sample_array[2]


# In[57]:


sample_array[0]


# In[59]:


sample_array[2:6]


# In[60]:


sample_array[1:4]=100
sample_array


# In[61]:


sample_array=np.arange(10,21)
sample_array


# In[62]:


sample_array[0:7]


# In[63]:


subset_sample_array=sample_array[0:7]
subset_sample_array


# In[64]:


subset_sample_array[:]=1001
subset_sample_array


# # two dimentional array

# In[65]:


import numpy as np


# In[67]:


sample_matrix=np.array([[50,20,1,23],[24,23,56,76],[62,95,0,10]])
sample_matrix


# In[68]:


sample_matrix[1,2]


# In[69]:


sample_matrix[2,:]


# In[71]:


sample_matrix


# In[72]:


sample_matrix[2]


# In[73]:


sample_matrix[:,(3,2)]


# In[75]:


sample_matrix[(2,2),(2,3)]


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




