#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb


# In[2]:


from sklearn.datasets import load_breast_cancer;


# In[3]:


breast_cancer = load_breast_cancer();


# In[4]:


X = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)


# In[5]:


X


# In[6]:


y = pd.DataFrame(breast_cancer.target)


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[9]:


from sklearn.tree import DecisionTreeClassifier;


# In[10]:


dtree = DecisionTreeClassifier(max_depth=4)


# In[11]:


dtree.fit(X_train,y_train)


# In[12]:


predictions = dtree.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix


# In[17]:


print(confusion_matrix(predictions,y_test))
print('\n')
print(classification_report(predictions,y_test))


# In[ ]:





# In[ ]:





# # Adopting XGBoost

# In[20]:


# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))


# In[23]:


# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])


# In[ ]:




