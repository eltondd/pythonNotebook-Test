#!/usr/bin/env python
# coding: utf-8

# # Test Notebook

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# In[13]:


dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')


# In[14]:


y = data.quality
X = data.drop(columns=['pH', 'sulphates'])
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.8, 
                                                    random_state=123, 
                                                    stratify=y)


# In[2]:


print('hello_world')
print('ok....')


# In[15]:


plt.plot(X_train, y_train)
plt.show()


# In[16]:


print('{}, {}'.format(X_train.shape, y_train.shape))

