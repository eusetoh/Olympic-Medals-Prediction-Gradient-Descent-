#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns


# In[4]:


#read data, data from https://github.com/dataquestio/project-walkthroughs/tree/master/linear_regression
teams = pd.read_csv("C:/Users/Eusebius/Desktop/Self-Initiated Projects/Olympic Medal Mahine Learning Python/teams.csv")
teams


# In[11]:


#select only relevant columns as new table
teams1 = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
teams1


# In[9]:


#check for and remove missing data (listwise)
teams1[teams1.isnull().any(axis=1)]
teams2 = teams1.dropna()


# In[10]:


#visalise clean dataset
teams2


# In[12]:


#explore correlations - note high correlation between prev_medals (IV) and medals (DV)
teams2.corr(numeric_only = True)["medals"]


# In[13]:


#visualise correlation between athletes and medals
sns.lmplot(x="prev_medals", y="medals", data=teams2, fit_reg=True)


# In[20]:


#train test split by year 2012
train = teams2[teams2["year"]< 2012].copy()
test = teams2[teams2["year"] >= 2012].copy()

train.shape
test.shape


# In[47]:


#fit the linear regression model with training data
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(train[["prev_medals"]], train[["medals"]])

print("Coefficient:", reg.coef_)
print("Intercept:", reg.intercept_)


# In[28]:


#evaluate the model based on test data
predictions = reg.predict(test[["prev_medals"]])

from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test["medals"],predictions)
error


# In[41]:


#prep data for gradient descent
x = teams2[["prev_medals"]]
y = teams2["medals"]
y = y.to_numpy().flatten()
x = x.to_numpy().flatten()


# In[53]:


#gradient descent
import numpy as np
import matplotlib.pyplot as plt

# general function y = wx + b
#loss function = MSE = (predicted y - y)**2

general_loss = lambda x,y,w,b: ((w @ x + b) - y) **2 #predicted y -y squared

#set random value for w and b - 0.5 and 1
w = 0.5
b = 1

def forward (x):
    prediction = x * w + b
    return prediction

def mse (y, prediction):
    return np.mean((prediction - y)**2)

predictions = forward (x)
mse(y, predictions)

def mse_grad (y, predicted):
    grad_w = np.mean(x *(predicted - y)*2)
    grad_b = np.mean((predicted - y)*2)
    return grad_w, grad_b

lr = 0.0001
epochs = 10000

for i in range(epochs):
    predictions = forward (x)
    loss = mse (y, predictions)
    grad_w = mse_grad(y, predictions)[0]
    grad_b = mse_grad(y, predictions)[1]
    w = w - grad_w * lr
    b = b - grad_b * lr
    
    if i % 1000 == 0:
        print (f"Epoch {i} loss: {loss}")
    
print(f"Medals = {b:.4f}: + {w:.4f}: * prev_medals") 


# In[54]:


#comparing output from Linear Regression and Gradient Descent
print(f"Linear Regression: Medals = {reg.intercept_.item():.4f} + {reg.coef_[0, 0]:.4f} * prev_medals")
print(f"Gradient Descent: Medals = {b:.4f} + {w:.4f} * prev_medals") 

