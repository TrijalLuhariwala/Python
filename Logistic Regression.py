#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[73]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


df=pd.read_csv("titanic-training-data.csv")


# In[75]:


df.shape


# In[76]:


df.head()


# In[77]:


df.dtypes


# In[78]:


df.info()


# In[79]:


df.isnull().sum()


# ### EDA

# In[80]:


### Analyze dependent variable
sns.countplot(x="Survived",data=df,palette="Paired")


# In[81]:


sns.countplot(x="Survived",hue="Sex",data=df)


# In[82]:


sns.countplot(x="Survived",hue="Pclass",data=df)


# In[83]:


df.drop(["PassengerId","Name","Ticket","Fare","Cabin"],axis=1,inplace=True)
df.head()


# In[84]:


df.hist(figsize=(20,30))
plt.show()


# In[85]:


sns.countplot(x="SibSp",data=df)


# In[86]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")


# In[87]:


sns.boxplot(x="Pclass",y="Age",data=df)


# In[88]:


df.dropna(inplace=True)   ## Removing null values


# In[89]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")


# In[90]:


df.isnull().sum()


# In[91]:


df=pd.get_dummies(df,columns=["Pclass","Sex","Embarked"])


# In[92]:


df.isnull().sum()


# In[93]:


df.dtypes


# In[94]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[95]:


X=df.drop(['Survived'],axis=1)
y=df[['Survived']]


# In[96]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)


# In[ ]:





# ## Fit the model

# In[97]:


import warnings 
warnings.filterwarnings("ignore")


# In[98]:


model=LogisticRegression(solver="lbfgs")
model.fit(X_train,y_train)
model


# In[99]:


predictions=model.predict(X_test)


# In[100]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# In[101]:


model.score(X_train,y_train)


# In[102]:


model.score(X_test,y_test)


# In[103]:


predictions=model.predict(X_test)


# In[104]:


from sklearn import metrics


# In[105]:


print(metrics.classification_report(y_test,predictions))


# In[106]:


cm=metrics.confusion_matrix(y_test,predictions,labels=[1,0])
df_cm=pd.DataFrame(cm,index=[i for i in ["1","0"]],
                  columns=[i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[ ]:




