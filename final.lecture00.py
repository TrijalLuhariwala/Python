#!/usr/bin/env python
# coding: utf-8

# # <div style="color:red;background-color:lime;padding:3%;border-radius:150px150px;font-size:2em;text-align:center">Data analysis with-LR-DT-RF-and-SVM-99.6% AUC</div>

# In[1]:


## <div style="color:red;background-color:lime;padding:3%;border-radius:150px150px;font-size:2em;text-align:center">Data analysis with-LR-DT-RF-and-SVM-99.6% AUC</div>


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import matplotlib
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
matplotlib.rcParams.update({'font.size':15})


# In[ ]:





# In[3]:


plt.style.use('dark_background')


# In[4]:


df=pd.read_csv("predictive_maintenance.csv")


# In[5]:


df.sample(10).style.set_properties(
    **{
        'background-color':'cyan',
        'color':'black',
        'border-color':'magenta'
    })


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


## Converting kelv into degree celcius temperature substracting 273.15

df["Air temperature [K]"]=df["Air temperature [K]"]-273.15
df["Process temperature [K]"]=df["Process temperature [K]"]-273.15
df.rename(columns={"Air temperature [K]":"Air temperature [C]","Process temperature [K]":"Process temperature [C]"},inplace=True)


# In[9]:


df.sample()


# In[10]:


df["Temperature difference"]=df["Process temperature [C]"]-df["Air temperature [C]"]
df.sample(5)


# In[11]:


df=df.drop(["UDI","Product ID"],axis=1)


# In[12]:


## CHeck for missing values and write code for describe 


# In[13]:


df.describe().style.background_gradient(cmap="magma")


# In[14]:


df.isnull().sum()


# In[15]:


## No missing values so no replacement required


# In[16]:


df.dtypes


# In[17]:


import missingno as msno
msno.matrix(df,color=(1,0.8,0.6))


# In[18]:


sns.displot(data=df,x="Air temperature [C]",kde=True,bins=100,facecolor="cyan",color="blue",height=5,aspect=3)


# In[19]:


sns.displot(data=df,x="Process temperature [C]",kde=True,bins=100,facecolor="grey",color="blue",height=5,aspect=3)


# In[20]:


sns.displot(data=df,x="Temperature difference",kde=True,bins=100,facecolor="cyan",color="blue",height=5,aspect=3)


# In[21]:


sns.countplot(x="Type",data=df)
plt.title("Title",color="red",font="Times New Roman")


# In[22]:


df["Type"].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("Type")


# In[23]:


ax=plt.figure(figsize=(18,9))
ax=plt.subplot(1,2,1)
sns.countplot(x="Type",data=df)
plt.title("Title",color="red",font="Times New Roman")
ax=plt.subplot(1,2,2)
df["Type"].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("Type")


# In[24]:


ax=plt.figure(figsize=(18,9))
ax=plt.subplot(1,2,1)
sns.countplot(x="Target",data=df)
plt.title("Target",color="red",font="Times New Roman")
ax=plt.subplot(1,2,2)
df["Target"].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("Target")


# In[25]:


sns.scatterplot(data=df,x="Torque [Nm]",y="Rotational speed [rpm]",hue="Failure Type",palette="tab10")


# In[26]:


sns.scatterplot(data=df,x="Torque [Nm]",y="Rotational speed [rpm]",hue="Target",palette="tab10")


# In[27]:


df.dtypes


# ### Label encoding

# In[28]:


from sklearn.preprocessing import LabelEncoder
st=LabelEncoder()
df["Failure Type"]=st.fit_transform(df["Failure Type"])
df["Type"]=st.fit_transform(df["Type"])


# In[29]:


df.dtypes


# In[32]:


X=df.drop(columns="Failure Type",axis=1)
y=df["Failure Type"]


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)


# In[34]:


from sklearn.svm import SVC ## Use when the pairplot contains clumsy data and some organised data otherwise use linear regression
svc=SVC()
svc.fit(X_train,y_train)


# In[35]:


svc.score(X_train,y_train)


# In[36]:


svc.score(X_test,y_test)


# In[ ]:




