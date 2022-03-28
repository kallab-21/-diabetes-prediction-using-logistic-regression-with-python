#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# BUILDING LOGISTIC REGRESSION FROM SCRATCH


# In[49]:


import pandas as pd
get_ipython().system('pip install numpy')
import numpy as np


# In[5]:


get_ipython().system('pip3 install -U scikit-learn scipy matplotlib')
from sklearn.preprocessing import StandardScaler


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


# In[19]:


diabetes_dataset = pd.read_csv('C:\\Users\kal ab\Desktop\ML\\diabetes_binary_5050split_health_indicators_BRFSS2015.csv') 


# In[20]:


diabetes_dataset.head()


# In[21]:


diabetes_dataset.shape


# In[22]:


diabetes_dataset.describe()


# In[25]:


diabetes_dataset['Diabetes_binary'].value_counts()


# In[27]:


diabetes_dataset.groupby('Diabetes_binary').mean()


# In[28]:


features = diabetes_dataset.drop(columns = 'Diabetes_binary', axis=1)
target = diabetes_dataset['Diabetes_binary']


# In[29]:


print(features)


# In[ ]:





# In[30]:


print(target)


# In[34]:


scaler=StandardScaler()
features=scaler.fit_transform(features)


# In[33]:


scaler.fit(features)


# In[38]:


X_train, X_test, Y_train,Y_test = train_test_split(features,target, test_size = 0.2, random_state=2)


# In[36]:


print(features.shape, X_train.shape, X_test.shape)


# In[ ]:


#


# In[70]:


class Logistic_Regression():


  # declaring learning rate & number of iterations (Hyperparametes)
  def __init__(self, learning_rate, no_of_iterations):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations



  # fit function to train the model with dataset
  def fit(self, X, Y):

    # number of data points in the dataset (number of rows)  -->  m
    # number of input features in the dataset (number of columns)  --> n
    self.m, self.n = X.shape


    #initiating weight & bias value

    self.w = np.zeros(self.n)
    
    self.b = 0

    self.X = X

    self.Y = Y


    # implementing Gradient Descent for Optimization

    for i in range(self.no_of_iterations):     
      self.update_weights()



  def update_weights(self):

    # Y_hat formula (sigmoid function)

    Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b ) ))    


    # derivaties

    dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))

    db = (1/self.m)*np.sum(Y_hat - self.Y)


    # updating the weights & bias using gradient descent

    self.w = self.w - self.learning_rate * dw

    self.b = self.b - self.learning_rate * db


  # Sigmoid Equation & Decision Boundary

  def predict(self, X):

    Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) ))     
    Y_pred = np.where( Y_pred > 0.5, 1, 0)
    return Y_pred


# In[73]:


Model = Logistic_Regression(learning_rate=0.003, no_of_iterations=100000)


# In[74]:


Model.fit(X_train, Y_train)


# In[75]:


X_train_prediction = Model.predict(X_train)
training_data_accuracy = accuracy_score( Y_train, X_train_prediction)


# In[76]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[78]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score( Y_test, X_test_prediction)


# In[79]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[83]:


input_data = (1.0,1.0,1.0,32.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,3.0,0.0,0.0,0.0,1.0,10.0,6.0,8.0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




