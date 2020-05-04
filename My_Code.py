#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the librarires
import numpy as np 
import  pandas as pd


# In[2]:


pwd #to check the working directory


# In[3]:


df=pd.read_csv("Churn_Modelling.csv")


# In[4]:


#showing the first ten rows of the dataset
df.head(10)


# In[5]:


#checking the number of columns
df.columns


# In[6]:


#checking the number of rows and columns
df.shape


# # We have 10000 Rows and 14 Columns
# 

# In[7]:


df.describe()


# # Checking For Null Missing Values

# In[8]:



df.isna().sum()


# # Checking the number of male and female who has exited the Bank
# 

# In[9]:


df.groupby("Gender")["Exited"].sum()


# # Showing the total number of customers Country Wise

# In[10]:


df.groupby(["Geography"])["Exited"].count()


# # Showing the Customers of France Who have exited the bank

# In[11]:


df[df["Geography"]=="France"]["Exited"].sum()


# In[12]:


Ratio1=df[df["Geography"]=="France"]["Exited"].sum()/df[df["Geography"]=="France"]["Exited"].count()
print(Ratio1)


# # We Can See that about 16% have leave the bank

# In[13]:


Ratio2=df[df["Geography"]=="Germany"]["Exited"].sum()/df[df["Geography"]=="Germany"]["Exited"].count()
print(Ratio2)


# In[14]:


Ratio3=df[df["Geography"]=="Spain"]["Exited"].sum()/df[df["Geography"]=="Spain"]["Exited"].count()
print(Ratio3)


# # We Can See that about 33% customers of Germany have left the Bank

# In[15]:


#showing the people
df.groupby("HasCrCard")["HasCrCard"].count()


# In[16]:


df.groupby("HasCrCard")["Exited"].sum()


# # WE Have Anlysed Our Data Let's Print Correlation Matrix

# In[17]:


df.corr()


# In[18]:


#defining the X_set
X=df.iloc[:,3:-1].values


# In[19]:


X


# In[20]:


len(X[0])


# In[21]:


#We Can See that we have two categorical Columns
print(X[:,1:3])


# In[22]:


#two categotrical values
#we have to encode them
#Making two objecs of LableEncoder Class
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb1=LabelEncoder()
lb2=LabelEncoder()


# In[23]:


#created two objects of LabelEncoder because we have two categorical values
X[:,1]=lb1.fit_transform(X[:,1])


# In[24]:


X


# In[25]:


X[:,2]=lb2.fit_transform(X[:,2])


# In[26]:


X


# In[27]:


#applying one hotencoding because we have more than two categories in the country column
one_hot=OneHotEncoder(categorical_features=[1])


# In[28]:


X=one_hot.fit_transform(X).toarray()


# In[29]:


X


# In[30]:


len(X[0])
# we have two more columns because of those two more more countries that we have onehotencoded


# In[31]:


# to overcome the dummyvariable trap
X=X[:,1:]


# In[32]:


X


# In[33]:


len(X[0])


# In[34]:


y=df.iloc[:,-1].values


# # Splitting the Dataset to train and test 

# In[35]:



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

# Here I Have keep the test_size=0.2
# meaning the 20% will be test set and 80% training set
#random_state so that it will keep that state,meaning on recompling
#weights wont change


# In[36]:


#we have preprocessed the data 
#now its time to apply the algorithms
import tensorflow 


# # We Are Using Tensorflow

# In[37]:


import keras


# In[38]:


#For Defining the Layers
from tensorflow.keras.layers import Dense


# In[39]:


#For Defining the Model
from tensorflow.keras.models import Sequential


# In[40]:


#Created An Object of Model
regressor=Sequential()


# # Defining The Layers

# In[41]:


regressor.add(Dense(units=6,input_shape=(11,), activation='relu'))
#units means the Neurons on the Outer(Here Hidden Layer)


# In[42]:


regressor.add(Dense(units=6,activation='relu'))
#We need not to define the input_shape here because the now it is known 
#to the layer as it will be accepting 6 outputs from the Previous Layer


# In[43]:


regressor.add(Dense(units=1,activation='sigmoid'))
#output Layer as Binary output so using Activation Function as sigmoid


# In[44]:


regressor.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[45]:



regressor.fit(X_train,y_train,batch_size=32,epochs=25)


# #  We can see that we have achieved accuracy 72% in 25 epochs

# In[46]:


def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6,activation='relu',input_dim=11))
    
    classifier.add(Dense(units=6,activation='relu'))
    
    classifier.add(Dense(units=1,activation='sigmoid'))
    
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
                   


# In[57]:


y_pred=regressor.predict(X_test)
y_pred=(y_pred>0.5)
print(y_pred)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[58]:


from keras.wrappers.scikit_learn import KerasClassifier
classifier=KerasClassifier(build_fn=build_classifier)


# In[59]:



from sklearn.model_selection import GridSearchCV
parameters = {'epochs': [ 50,100], 'batch_size': [16,32]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[60]:


best_accuracy


# In[61]:


best_parameters


# In[ ]:





# In[ ]:





# In[ ]:




