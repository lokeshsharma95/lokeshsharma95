#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The code imports necessary libraries and modules for data manipulation, visualization, preprocessing, model selection, linear regression, and performance evaluation.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[6]:


#This code reads the house price dataset from a CSV file and stores it in a pandas DataFrame called customers. The .head() method is then used to display the first few rows of the DataFrame.
customers = pd.read_csv(r'C:\Users\sharm\OneDrive\Desktop\House_price.csv')
customers.head()


# In[7]:


#The describe() method provides statistical summary (count, mean, standard deviation, quartiles, etc.) of the numerical columns in the DataFrame, giving an overview of the dataset's distribution.
customers.describe()


# In[8]:


#The info() method provides a concise summary of the DataFrame, including the number of non-null values and the data types of each column.
customers.info()


# In[9]:


# This code generates a pair plot using seaborn (sns) to visualize the relationships between pairs of variables in the dataset.
sns.pairplot(customers)


# In[10]:


# A StandardScaler object is created for feature scaling. The input features (X) are obtained by dropping the 'YrSold' and 'SalePrice' columns from the customers DataFrame, while the target variable (y) is assigned the 'SalePrice' column.
scaler = StandardScaler()
X=customers.drop(['YrSold','SalePrice'],axis=1)
y=customers['SalePrice']

#The column names (features) of the DataFrame are stored in the cols variable. The feature values in X are then standardized (scaled) using the fit_transform() method of the StandardScaler object.
cols= X.columns
X= scaler.fit_transform(X)


# In[29]:


#The dataset is split into training and testing sets using the train_test_split() function. 70% of the data is used for training, while the remaining 30% is reserved for testing. The random_state parameter is set to 121 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=121)


# In[30]:


# A LinearRegression object (lr) is created, and the model is trained on the training data using the fit() method.
lr = LinearRegression()
lr.fit(X_train,y_train)

#The trained model is used to make predictions on the testing data (X_test), and the r2_score() function is used to evaluate the model's performance by calculating the coefficient of determination (R-squared) between the predicted values (pred) and the actual values (y_test).
pred= lr.predict(X_test)
r2_score(y_test,pred)


# In[31]:


# A scatter plot is generated using seaborn to visualize the predicted values (pred) against the actual values (y_test).
sns.scatterplot(x=y_test, y=pred)


# In[32]:


# A histogram plot is generated using seaborn to visualize the distribution of the residuals (the difference between the actual and predicted values).
sns.histplot((y_test-pred), bins=50,kde=True)


# In[36]:


# A DataFrame (cdf) is created to store the coefficients (weights) of the linear regression model, associated with each feature. The coefficients are sorted in descending order to show the most influential features.
cdf=pd.DataFrame(lr.coef_,cols,['coefficients']).sort_values('coefficients',ascending=False)
cdf


# In[ ]:




