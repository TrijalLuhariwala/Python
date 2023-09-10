#!/usr/bin/env python
# coding: utf-8

# # KNN Algorithm

# ### We will use KNN(K nearest neighbours) algorithm to predict the type of breast cancer in the breast.

# In[53]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore


# ## Load data

# In[54]:


df=pd.read_csv("wisc_bc_data.csv")


# In[55]:


df.shape


# In[56]:


df.dtypes


# In[57]:


df['diagnosis']=df.diagnosis.astype('category')


# In[58]:


df.describe().T  ## T for transpose so to have simplified statistics can also use .transpose()


# In[59]:


df.groupby(["diagnosis"]).count()
##  B or M is to be predicted which are the types of breast cancer. Class distribution among B and M is almost 2:1 The model will better predict B and M


# ### drop first column from the data frame. this is id column which is not used in modeling

# In[60]:


## The first column which is  patient id and nothing to do with the model attributes.So drop the id colum
df=df.drop(labels="id",axis=1)


# In[61]:


df.shape


# In[62]:


df.isnull().sum()


# In[63]:


## Create a separate dataframe consisting only of the features i.e. independent attributes

X=df.drop(labels="diagnosis",axis=1)
y=df["diagnosis"]
X.head()


# In[64]:


## convert the features into z scores as we do not know what units/scales were used and store them in new
## it is always adviced to scale numeric attributes in model that calculate distance

XScaled=X.apply(zscore)  #convert all atributes to Z scale

XScaled.describe()


# In[65]:


# Split X and y into training and test set in 75:25 ratio

X_train,X_test,y_train,y_test=train_test_split(XScaled,y,test_size=0.30,random_state=1)


# ## Build KNN model

# In[66]:


model_2=KNeighborsClassifier(n_neighbors=5,weights='distance')


# In[67]:


## Call Nearest Neighbors algorithm
model_2.fit(X_train,y_train)


# ## Evaluate Performance of KNN model

# In[68]:


## For every data point,predict its label based on 5 nearest neighbors in this model. The majority 
## be assigned to the test data point 

predicted_labels =model_2.predict(X_test)
model_2.score(X_test,y_test)


# In[69]:


model_2.score(X_train,y_train)


# In[ ]:


## Calculate the accuracy measures and conusion matrix
from sklearn import metrics

print("Confusion Matrix")
cm=metrics.confusion_matrix(y_test,predicted_labels,labels=["M","B"])

df_cm=pd.DataFrame(cm,index=[i for i in ["M","B"]],
                  columns=[i for i in ["PredictM","Predict B"]])


# In[ ]:





# In[ ]:




