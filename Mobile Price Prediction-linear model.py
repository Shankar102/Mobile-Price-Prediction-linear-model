#!/usr/bin/env python
# coding: utf-8

# Recently you entered in the mobile manufacturing market. 
# Build a machine learning model that would help you to know the estimated price for
# your manufactured mobile phones on the basis of various key features and specifications.

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
import seaborn as sns
print("Done")


# In[69]:


df=pd.read_csv('data/mobile_price_data.csv')
df.head(5)


# In[57]:


df.describe()


# In[58]:


df.info


# In[72]:


df['m_price'].value_counts


# In[71]:


df['m_weight'].value_counts


# In[ ]:





# In[70]:


sns.lmplot(x='m_price',y='m_weight',data=df,order=2,ci=None)


# In[75]:


plt.figure(figsize=(15,6))
sns.countplot('m_price',data=df.head(100))
plt.xticks(rotation=90)
plt.show()


# In[76]:


plt.figure(figsize=(15,6))
sns.countplot('m_weight',data=df.head(100))
plt.xticks(rotation=90)
plt.show()


# In[90]:


X=np.array(df['m_price']).reshape(-1,1)
y=np.array(df['m_weight']).reshape(-1,1)

df.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

lg=LinearRegression() 
lg.fit(X_train,y_train)
print(lg.score(X_test,y_test))


# In[91]:


df.fillna(method='ffill',inplace=True)


# In[92]:


y_pred=lg.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[93]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
rmse=mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
print("mae",mae)
print("mse",mse)
print("rmse",rmse)


# In[99]:


df_binary500=df[:][:500]

sns.lmplot(x="m_price",y="m_weight",data=df_binary500,order=1000,ci=None)


# In[100]:


X=np.array(df['m_price']).reshape(-1,1)
y=np.array(df['m_weight']).reshape(-1,1)

df.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

lg=LinearRegression() 
lg.fit(X_train,y_train)
print(lg.score(X_test,y_test))


# In[101]:


y_pred=lg.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[ ]:




