#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import HTML
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import scipy.stats

import statsmodels.api as sm
import json
import time
import pylab
from scipy import stats
from datetime import date
import datetime as dt

import plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

from IPython.display import display, Math, Latex

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


# # The data is available as two attached CSV files:
# 
# takehome_user_engagement.csv
# takehome_users.csv
# The data has the following two tables:
# 1. User table
# ( "takehome_users" ) with data on 12,000 users who signed up for the product in the last two years.
# This table includes:
# 
# ● name: the user's name
# 
# ● object_id: the user's id
# 
# ● email: email address
# 
# ● creation_source: how their account was created. 
# This takes on one of 5 values:
# 
# PERSONAL_PROJECTS: invited to join another user's personal workspace
# 
# GUEST_INVITE: invited to an organization as a guest (limited permissions)
# ORG_INVITE: invited to an organization (as a full member)
# SIGNUP: signed up via the website
# SIGNUP_GOOGLE_AUTH: signed up using Google Authentication (using a Google email account for their login id)
# 
# a. creation_time: when they created their account
# 
# b. last_session_creation_time: unix timestamp of last login
# 
# c. opted_in_to_mailing_list: whether they have opted into receiving marketing emails
# 
# d. enabled_for_marketing_drip: whether they are on the regular marketing email drip
# 
# e. org_id: the organization (group of users) they belong to
# 
# f. invited_by_user_id: which user invited them to join (if applicable).

# In[3]:


import os
os.chdir(r"C:\\Users\\shobi\\OneDrive\\Documents")


# In[4]:


takehome_users = pd.read_csv('takehome_users.csv',encoding='ISO-8859-1')
takehome_users.head()


# In[5]:


takehome_users.info()


# # 2. Usage summary table
# ( "takehome_user_engagement" ) that has a row for each day that a user logged into the product.

# In[6]:


takehome_user_engagement=pd.read_csv('takehome_user_engagement.csv')
takehome_user_engagement.head()


# In[7]:


takehome_user_engagement.info()


# # Defining an "adopted user" as a user who has logged into the product on three separate days in at least one seven day period, identify which factors predict future user adoption.
# 
# We suggest spending 1-2 hours on this, but you're welcome to spend more or less.
# 
# Please send us a brief write-up of your findings (the more concise, the better no more than one page), along with any summary tables, graphs, code, or queries that can help us understand your approach.
# 
# Please note any factors you considered or investigation you did, even if they did not pan out. Feel free to identify any further research or data you think would be valuable.

# In[8]:


#Functions I commonly use to deal with date/time values
def get_date_int(df, column):
    '''
    This handy function parses year,month,week,day.
    '''
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day

def get_week(x): return x.isocalendar()

def get_iso_date_int(df,column):
    '''
    With time coded as iso (year,week,day) this seperates those time periods.
    '''
    temp_df=pd.DataFrame(df[column].tolist(), index=df.index)
    year,week,day=temp_df[0],temp_df[1],temp_df[2]
    return year,week,day


# In[9]:


takehome_users = pd.read_csv('takehome_users.csv',encoding='ISO-8859-1')
#code creation_time,last_session_time as date/time
takehome_users.creation_time = pd.to_datetime(takehome_users['creation_time'])
takehome_users.last_session_creation_time = pd.to_datetime(takehome_users['last_session_creation_time'])
#change column heading
takehome_users['user_id'] = takehome_users['object_id']
#drop original column
takehome_users.drop('object_id', axis=1, inplace=True)
#drop private information
takehome_users.drop(['name', 'email'], axis=1, inplace=True)

takehome_users.head()


# In[10]:


takehome_users.info()


# In[11]:


takehome_user_engagement['time_stamp'] = pd.to_datetime(takehome_user_engagement['time_stamp'])

takehome_user_engagement['week_time_stamp']=takehome_user_engagement['time_stamp'].apply(get_week)


# In[12]:


print('First user engagement timestamp:',min(takehome_user_engagement.time_stamp))
print('Last user engagement timestamp:',max(takehome_user_engagement.time_stamp))


# In[13]:


year, month, day=get_date_int(takehome_user_engagement, 'time_stamp')
takehome_user_engagement['year'],takehome_user_engagement['month'],takehome_user_engagement['day']=year,month,day
takehome_user_engagement['week']=takehome_user_engagement['time_stamp'].dt.week
#Make year and week, So if we are dealing with 52 week units then I want year to make it individual unit of time
iso_year,iso_week,iso_day=get_iso_date_int(takehome_user_engagement,'week_time_stamp')
takehome_user_engagement['year_week']=list(zip(iso_year,iso_week))


# # Defining an "adopted user" as a user who has logged into the product on three separate days in at least one seven day period, identify which factors predict future user adoption.
# 
# a. After playing with the data and thinking about the problem I decided the easiest time scale to use is the year/week units I created. I minimized the data to values I will need to solve the problem of 'adopted users'.

# In[14]:


takehome_user_engagement=takehome_user_engagement.sort_values(['time_stamp','user_id'],ascending=True)
takehome_user_engagement=takehome_user_engagement[['user_id','visited','day','year_week']]


# In[ ]:


adopted_user_dict={}
weeks=takehome_user_engagement.year_week
user_ids=list(set(takehome_user_engagement['user_id']))

for i in range(len(user_ids)):
    user_id=user_ids[i]
    reduced_df=takehome_user_engagement[(takehome_user_engagement['user_id']==user_id)&(weeks.isin(weeks[weeks.duplicated()]))]
    week_counts=reduced_df.year_week.value_counts()[reduced_df.year_week.value_counts()>2]
    three_logins=reduced_df[reduced_df.year_week.isin(list(week_counts.index))]
    three_logins=three_logins[~three_logins.duplicated()]
    adopted_user_dict[str(user_id)]=len(three_logins)


# In[ ]:


takehome_user_engagement['engagement_index']=takehome_user_engagement['user_id'].apply(lambda x: adopted_user_dict[str(x)])
takehome_user_engagement['adopted_user']=0
takehome_user_engagement['adopted_user'][takehome_user_engagement['engagement_index']>0]=1


# In[ ]:


adopted_count=takehome_user_engagement[['user_id','adopted_user']][takehome_user_engagement['adopted_user']==1].groupby('user_id').count()
print('Number of adopted users:',len(adopted_count))


# In[ ]:


adopted=takehome_user_engagement[['user_id','adopted_user']]
adopted_users = pd.merge(takehome_users, adopted, on='user_id', how='outer')


# In[ ]:


creation_year, creation_month, creation_day=get_date_int(adopted_users, 'creation_time')
last_session_year, last_session_month, last_session_day=get_date_int(adopted_users, 'last_session_creation_time')
adopted_users['creation_year'],adopted_users['creation_month'],adopted_users['creation_day']=creation_year, creation_month, creation_day
adopted_users['last_session_year'],adopted_users['last_session_month'],adopted_users['last_session_day']=last_session_year, last_session_month, last_session_day
adopted_users.drop(['creation_time', 'last_session_creation_time', 'user_id'], axis=1, inplace=True)


# In[ ]:


adopted_users.last_session_day.fillna(0, inplace=True)
adopted_users.last_session_month.fillna(0, inplace=True)
adopted_users.last_session_year.fillna(0, inplace=True)|


# In[ ]:


from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
le = preprocessing.LabelEncoder()
adopted_users['creation_source']=le.fit_transform(adopted_users['creation_source'])


# In[ ]:


adopted_users['invited'] = np.where(adopted_users['invited_by_user_id'].isnull(), 1, 0)
adopted_users.drop('invited_by_user_id', axis=1, inplace=True)
#Fill in the missings
adopted_users=adopted_users.fillna(0)
#Create column labels for output
col_names=list(pd.Series(adopted_users.columns)[pd.Series(adopted_users.columns)!='adopted_user'])
#Code as arrays
X=adopted_users[list(pd.Series(adopted_users.columns)[pd.Series(adopted_users.columns)!='adopted_user'])].values
y=adopted_users['adopted_user'].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=3)
print('Train size:',(len(X_train)/len(X))*100)
print('Train observations:',(len(X_train)))
print('Test size:',(len(X_test)/len(X))*100)
print('Test observations:',(len(X_test)))


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=20,random_state=0,criterion='gini', class_weight='balanced')

clf.fit(X_train, y_train.ravel())
Accuracy=clf.score(X_train, y_train.ravel())
print('Accuracy:',Accuracy,'\n')

importFeature = clf.feature_importances_
feature_importances=pd.DataFrame([importFeature])

std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importFeature)[::-1]

# Print the feature ranking
print("Feature ranking:")

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importFeature[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

feature_importances=pd.DataFrame(pd.Series(col_names)[indices])
feature_importances['importance']=np.sort(importFeature)[::-1]
feature_importances.columns=['features','importance']
feature_importances


# In[ ]:




