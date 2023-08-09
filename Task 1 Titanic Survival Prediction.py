#!/usr/bin/env python
# coding: utf-8

# # Importing the lybraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ## Data Collection & Processing

# In[2]:


# load the data from csv file

titanic_data=pd.read_csv("tested.csv")


# In[3]:


# Print data

titanic_data.head()


# In[4]:


# Total number of rows & Columns
   
titanic_data.shape


# In[5]:


# Some informations about the data

titanic_data.info()


# In[6]:


# Missing value 

titanic_data.isnull().sum()


# In[7]:


#Handling the missing values
# drop cabin table 

titanic_data=titanic_data.drop(columns='Cabin',axis=1)


# In[8]:


#Replacing the missing values in "Age" column with mean value of age column

titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[9]:


# Search the mode value of "Embarked" column

print(titanic_data['Embarked'].mode())


# In[10]:


print(titanic_data['Embarked'].mode()[0])


# In[11]:


#Replacing the missing values in "Embarked" column with mode values

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[12]:


#Replacing the missing values in "Fare" column with mean values of fare column

titanic_data['Fare'].fillna(titanic_data['Fare'].mean(),inplace=True)


# In[13]:


#After filling missing values check again the number of missing values in each column

titanic_data.isnull().sum()


# In[14]:


#Getting some statistical information about the data

titanic_data.describe()


# ## Data Analysis

# In[15]:


#Getting some statistical information about the data

titanic_data.describe()


# ## Data Visualization

# In[16]:


sns.set()


# In[17]:


#Making a count plot for "Survived" column

sns.countplot(x='Survived', data=titanic_data)


# In[19]:


titanic_data['Sex'].value_counts()


# In[20]:


#Making a count plot for "Sex" column

sns.countplot(x='Sex', data=titanic_data)


# In[21]:


#Number of survivors Gender wise

sns.countplot(x='Sex', hue='Survived',data=titanic_data)


# In[22]:


# making a count plot for "Pclass" column

sns.countplot(x='Pclass', data=titanic_data)


# In[23]:


sns.countplot(x='Pclass', hue='Survived', data=titanic_data)


# In[24]:


titanic_data['Sex'].value_counts()


# In[25]:


titanic_data['Embarked'].value_counts()


# In[26]:


#Converting categorical Columns

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[27]:


titanic_data.head()


# In[28]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[29]:


print(X)


# In[30]:


print(Y)


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[32]:


print(X.shape, X_train.shape, X_test.shape)


# In[33]:


model = LogisticRegression()


# In[34]:


#Training the Logistic Regression model with training data

model.fit(X_train, Y_train)


# In[35]:


# accuracy on training data

X_train_prediction = model.predict(X_train)


# In[36]:


print(X_train_prediction)


# In[37]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[38]:


# accuracy on test data

X_test_prediction = model.predict(X_test)


# In[39]:


print(X_test_prediction)


# In[40]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

