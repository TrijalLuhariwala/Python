#!/usr/bin/env python
# coding: utf-8

# # Data structures

# # Lists

# In[1]:


sample_list=[1,13,15,"hoi",2]
sample_list


# In[2]:


type (sample_list)


# In[3]:


sample_list[1]


# In[4]:


sample_list[3]


# In[5]:


sample_list[5]


# 

# In[6]:


sample_list[-2]


# In[7]:


sample_list[0]


# In[8]:


sample_list


# # tuple

# # heterogeneous
# #### paranthesis
# #### can retrieve using index
# #### immutable

# In[9]:


sample_tuple=(1,23,45,"hiii",67,2)
sample_tuple


# In[10]:


sample_tuple[1]


# In[11]:


sample_tuple[3]


# In[12]:


sample_tuple[0]


# In[13]:


sample_tuple[-3]


# In[ ]:





# In[16]:


sapmle_tuple[10]


# In[17]:


sample_tuple[0]


# # Set

# # set is mutable1

# # does not allow duplicates
# ## ordered first placed
# ## elements cannot be retrieved using index

# In[ ]:





# In[19]:


sample_set={2,3,122,2,345,"hii",56,"apple","kite",4}
sample_set


# In[20]:


sample_set[0]


# In[21]:


sample_set[0]="hii"


# In[26]:


sample_set.add(101)


# 

# In[27]:


sample_set


# In[34]:


sample_set.remove(4)


# In[35]:


sample_set


# In[36]:


sample_set.add("apple")


# In[37]:


sample_set


# # Dictonary

# # has key value pair data structure
# #### key is unique
# #### values can be duplicated
# #### values can be retrieved using key
# #### key cannot be retrieved using value

# In[38]:


sample_dict={1:"apple",2:44,3:34.5,4:"hii",5:66,6:3536,7:"erer"}
sample_dict


# In[39]:


sample_dict[3]


# In[40]:


sample_dict["apple"]


# In[41]:


sample_dict[10]


# In[42]:


sample_dict[7]=22


# In[43]:


sample_dict


# In[44]:


sample_dict[11]="hello"


# In[45]:


sample_dict


# In[ ]:




