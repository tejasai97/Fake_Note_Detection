
# coding: utf-8



# In[1]:

import pandas as pd


# In[3]:

data = pd.read_csv('bank_note_data.csv')


# In[61]:

data.head()


# ## EDA

# In[67]:

import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[68]:

sns.countplot(x='Class',data=data)



# In[69]:

sns.pairplot(data,hue='Class')


# ## Data Preparation 


# In[71]:

from sklearn.preprocessing import StandardScaler



# In[72]:

scaler = StandardScaler()



# In[73]:

scaler.fit(data.drop('Class',axis=1))



# In[74]:

scaled_features = scaler.fit_transform(data.drop('Class',axis=1))



# In[77]:

df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# ## Train Test Split
# 

# In[79]:

X = df_feat


# In[80]:

y = data['Class']



# In[81]:

X = X.as_matrix()
y = y.as_matrix()



# In[45]:

from sklearn.cross_validation import train_test_split


# In[46]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# In[82]:

import tensorflow.contrib.learn.python.learn as learn



# In[83]:

classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)



# In[94]:

classifier.fit(X_train, y_train, steps=200, batch_size=20)


# ## Model Evaluation

# In[95]:

note_predictions = classifier.predict(X_test)


# In[96]:

from sklearn.metrics import classification_report,confusion_matrix


# In[97]:

print(confusion_matrix(y_test,note_predictions))


# In[98]:

print(classification_report(y_test,note_predictions))


# ##  Comparison

# In[99]:

from sklearn.ensemble import RandomForestClassifier


# In[100]:

rfc = RandomForestClassifier(n_estimators=200)


# In[101]:

rfc.fit(X_train,y_train)


# In[102]:

rfc_preds = rfc.predict(X_test)


# In[103]:

print(classification_report(y_test,rfc_preds))


# In[104]:

print(confusion_matrix(y_test,rfc_preds))


