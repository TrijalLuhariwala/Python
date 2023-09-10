#!/usr/bin/env python
# coding: utf-8

# ### Problem statement- wheather hotel booking is canceled or not

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("hotel_bookings.csv")


# In[3]:


df.sample(10)


# In[4]:


df.describe(include="all")


# In[5]:


df.isnull().sum()


# In[6]:


df.drop("company",axis=1,inplace=True)


# In[7]:


df.dtypes


# In[8]:


df.dtypes


# In[9]:


df.describe(include="all")


# ### Treating missing values

# In[10]:


mean1=df["agent"].mean()
df["agent"]=df["agent"].fillna(mean1)


# In[11]:


mode1=df["country"].mode()
df["country"]=df["country"].fillna(mode1)


# In[12]:


median1=df["is_canceled"].median()
median1


# In[13]:


df["is_canceled"].replace(np.NaN,median1,inplace=True)


# In[14]:


df.dropna(inplace=True)


# In[15]:


df=pd.get_dummies(df,columns=["hotel","arrival_date_month","meal","country","market_segment","distribution_channel","reserved_room_type","assigned_room_type","deposit_type","customer_type","reservation_status","reservation_status_date"])


# In[16]:


df.dtypes


# # EDA

# In[17]:


sns.countplot(x="is_canceled",data=df)


# In[18]:


sns.countplot(x="is_canceled",data=df,hue="arrival_date_year")


# In[19]:


sns.countplot(x="is_canceled",data=df,hue="arrival_date_week_number")


# In[20]:


sns.countplot(x="is_canceled",data=df,hue="children")


# In[21]:


sns.countplot(x="is_canceled",data=df,hue="babies")


# In[22]:


sns.countplot(x="is_canceled",data=df,hue="is_repeated_guest")


# In[23]:


sns.countplot(x="is_canceled",data=df,hue="previous_cancellations")


# In[24]:


sns.countplot(x="is_canceled",data=df,hue="booking_changes")


# In[25]:


sns.countplot(x="is_canceled",data=df,hue="required_car_parking_spaces")


# In[26]:


sns.boxplot(x="is_canceled",data=df)


# In[27]:


## sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")  


# In[28]:


## NO outliers therefore no need for outlier treatment


# In[29]:


## Fit the model in logistic regression in 70 30 ratio


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[31]:


X=df.drop(['is_canceled'],axis=1)
y=df[['is_canceled']]


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)


# In[33]:


import warnings 
warnings.filterwarnings("ignore")


# In[34]:


df.isnull().sum()


# In[35]:


model=LogisticRegression(solver="lbfgs")
model.fit(X_train,y_train)
model


# In[36]:


predictions=model.predict(X_test)


# In[37]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# In[38]:


model.score(X_train,y_train)


# In[39]:


model.score(X_test,y_test)


# In[40]:


from sklearn import metrics


# In[41]:


print(metrics.classification_report(y_test,predictions))


# In[42]:


cm=metrics.confusion_matrix(y_test,predictions,labels=[1,0])
df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],
                  columns=[i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:




