#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import dtale
import dtale.global_state as global_state


# In[2]:


df=pd.read_csv('COVID19_Data (2).csv')
df1=df.copy()


# In[3]:


df


# # Data Overview using D-tale

# In[4]:


global_state.set_chart_settings({'scatter_points': 75893, '3d_points': 40000})
d = dtale.show(df)
d.open_browser()


# # Manual Data Overview

# In[5]:


df.shape


# In[6]:


df.info()


# ### Convert numerical features to categorical 

# In[7]:


ls_cat=['facial_coverings','international_travel_controls','restrictions_internal_movements','close_public_transport',
        'stay_home_requirements','restriction_gatherings','cancel_public_events','workplace_closures','public_information_campaigns',
         'vaccination_policy','day of the week',]

for i in ls_cat:
    df[i]=df[i].astype('object')


# In[8]:


categorical_cols = df.select_dtypes(exclude = 'number').columns
numerical_cols = df.select_dtypes(include = 'number').columns


# In[9]:


for i in categorical_cols:
  print(i,"\n-------------\n",df[i].unique())
  print("number of unique values: ",len(df[i].unique()),"\n")


# ### Most common type of vaccine used per country 

# In[10]:


count_vacc=df.groupby('country')['vaccines_used'].value_counts().to_frame()
count_vacc


# # Visualizations

# In[11]:


def Visualizer(df,country,col1,col2):
    dff=df[df['country']==country]
    fig = px.scatter(dff, x="date", y=col1, hover_data=['country',col2],color=col2)
    fig.show()


# In[12]:


def summarize(df,col1,col2):
    frame1=df.groupby(col2)[col1].describe()
#     begin=[]
#     end=[]
#     for i in df.vaccination_policy.unique():
#         begin.append(df[df[col2]==i]['date'].tolist()[0])
#         end.append(df[df[col2]==i]['date'].tolist()[-1])
#     frame1['Start']=begin
#     frame1['End']=end
    return frame1


# In[13]:


# def regulations(df1, country ,regulation,col):
    
#     df=df1[df1['country']==country]
#     df[regulation]=df[regulation].astype('object')
#     fig = px.scatter(df, x='date', y=col, color=regulation, labels={regulation:'Regulation Label'},title=country)
#     fig.add_scatter()
#     fig.update_layout(xaxis_title='Date',)
#     fig.show()


# ### Containment Index

# In[14]:


quartiles=pd.qcut(df.containment_index, 3, labels=["Low", "Medium", "High"])
df['containment_index_quartiles']=quartiles


# In[15]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='containment_index_quartiles')
fig.show()


# In[16]:


summarize(df,'new_cases','containment_index_quartiles')


# In[17]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_deaths'],facet_col='containment_index_quartiles')
fig.show()


# In[18]:


summarize(df,'new_deaths','containment_index_quartiles')


# #### Containment Index In Jordan

# In[19]:


Visualizer(df,'Jordan','new_cases','containment_index') #Cases


# In[20]:


Visualizer(df,'Jordan','new_deaths','containment_index') #deaths


# ### Pollution Index 
# source: https://www.numbeo.com/pollution/rankings_by_country.jsp

# In[21]:


df1=pd.read_csv('pollution.csv')
df1=df1.rename(columns={'Country':'country'})
merged_df5= pd.merge(df, df1, on=['country'])


# In[22]:


quartiles=pd.qcut(merged_df5['Pollution Index'], 3, labels=["Low", "Medium", "High"])
merged_df5['poll_quar']=quartiles


# In[23]:


fig = px.scatter(merged_df5, x="date", y='new_deaths', hover_data=['country','new_deaths'],facet_col='poll_quar')
fig.show()


# In[24]:


summarize(merged_df5,'new_deaths','poll_quar')


# In[25]:


fig = px.scatter(merged_df5, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='poll_quar')
fig.show()


# In[26]:


summarize(merged_df5,'new_cases','poll_quar')


# Countries with "Low" Pollution index have the highest average number of cases and the highest standard deviation.

# ## Regulations VS  Cases VS  Deaths
# ### Workplace Closures
# 0: No restrictions <br>
# 1: Recommend closing (or work from home) <br>
# 2: Require closing (or work from home) for some sectors or categories of workers <br>
# 3: Require closing (or work from home) all but essential workplaces (e.g. grocery stores, doctors) <br>

# In[27]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='workplace_closures')
fig.show()


# In[28]:


summarize(df,'new_cases','workplace_closures')


# In[29]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_cases'],facet_col='workplace_closures')
fig.show()


# In[30]:


summarize(df,'new_deaths','workplace_closures')


# In[31]:


Visualizer(df,'Jordan','new_cases','workplace_closures')#Cases


# In[32]:


Visualizer(df,'Jordan','new_deaths','workplace_closures')#Deaths


# ### Cancelation Of Public Events

# 0- No measures <br>
# 1 - Recommend cancelling <br>
# 2 - Require cancelling <br>

# In[33]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='cancel_public_events')
fig.show()


# In[34]:


summarize(df,'new_cases','cancel_public_events')


# In[35]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_deaths'],facet_col='cancel_public_events')
fig.show()


# In[36]:


summarize(df,'new_deaths','cancel_public_events')


# In[37]:


Visualizer(df,'Jordan','new_cases','cancel_public_events') #Cases


# In[38]:


Visualizer(df,'Jordan','new_deaths','cancel_public_events') #Deaths


# ### Restrictions On Internal Movements

# 0 - No measures  <br>
# 1 - Recommend movement restriction  <br>
# 2 - Restrict movement <br>

# In[39]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='restrictions_internal_movements')
fig.show()


# In[40]:


summarize(df,'new_cases','restrictions_internal_movements')


# In[41]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_deaths'],facet_col='restrictions_internal_movements')
fig.show()


# In[42]:


summarize(df,'new_deaths','restrictions_internal_movements')


# In[43]:


Visualizer(df,'Jordan','new_cases','restrictions_internal_movements') #Cases


# In[44]:


Visualizer(df,'Jordan','new_deaths','restrictions_internal_movements') #Deaths


# ### Gathering Restrictions

# 0 - No restrictions <br>
# 1 - Restrictions on very large gatherings (the limit is above 1,000 people)  <br>
# 2 - Restrictions on gatherings between 100-1,000 people  <br>
# 3 - Restrictions on gatherings between 10-100 people  <br>
# 4 - Restrictions on gatherings of less than 10 people  <br>

# In[45]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='restriction_gatherings')
fig.show()


# In[46]:


summarize(df,'new_cases','restriction_gatherings')


# In[47]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_cases'],facet_col='restriction_gatherings')
fig.show()


# In[48]:


summarize(df,'new_deaths','restriction_gatherings')


# In[49]:


Visualizer(df,'Jordan','new_cases','restriction_gatherings') #Cases


# In[50]:


Visualizer(df,'Jordan','new_deaths','restriction_gatherings') #Deaths


# ### Stay At Home Requirements

# 0 - No measures <br>
# 1 - Recommend not leaving house <br>
# 2 - Require not leaving house with exceptions for daily exercise, grocery shopping, and ‘essential’ trips <br>
# 3 - Require not leaving house with minimal exceptions (e.g. allowed to leave only once every few days, or only one  person can leave at a time, etc.) <br>

# In[51]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='stay_home_requirements')
fig.show()


# In[52]:


summarize(df,'new_cases','stay_home_requirements')


# In[53]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_deaths'],facet_col='stay_home_requirements')
fig.show()


# In[54]:


summarize(df,'new_deaths','stay_home_requirements')


# In[55]:


Visualizer(df,'Jordan','new_cases','stay_home_requirements') #Cases


# In[56]:


Visualizer(df,'Jordan','new_deaths','stay_home_requirements') #Deaths


# ### Facial coverings

# 0- No policy <br>
# 1- Recommended <br>
# 2- Required in some specified shared/public spaces outside the home with other people present, or some situations when social distancing not possible <br>
# 3- Required in all shared/public spaces outside the home with other people present or all situations when social distancing not possible <br>
# 4- Required outside the home at all times, regardless of location or presence of other people <br>

# In[57]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='facial_coverings')
fig.show()


# In[58]:


summarize(df,'new_cases','facial_coverings')


# In[59]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_deaths'],facet_col='facial_coverings')
fig.show()


# In[60]:


summarize(df,'new_deaths','facial_coverings')


# In[61]:


Visualizer(df,'Jordan','new_cases','facial_coverings') #Cases


# In[62]:


Visualizer(df,'Jordan','new_deaths','facial_coverings') #Deaths


# ### Vaccination Policy ( Availability )

# 0 - No availability <br>
# 1 - Availability for ONE of the following: key workers/ clinically vulnerable groups / elderly groups <br>
# 2 - Availability for TWO of the following: key workers/ clinically vulnerable groups / elderly groups <br>
# 3 - Availability for ALL the following: key workers/ clinically vulnerable groups / elderly groups <br>
# 4 - Availability for all three, plus partial additional availability (select broad groups/ages) <br>
# 5 - Universal availability <br>

# In[63]:


fig = px.scatter(df, x="date", y='new_cases', hover_data=['country','new_cases'],facet_col='vaccination_policy')
fig.show()


# In[64]:


summarize(df,'new_cases','vaccination_policy')


# In[65]:


fig = px.scatter(df, x="date", y='new_deaths', hover_data=['country','new_deaths'],facet_col='vaccination_policy')
fig.show()


# In[66]:


summarize(df,'new_deaths','vaccination_policy')


# In[67]:


Visualizer(df,'Jordan','new_cases','vaccination_policy') #Cases


# In[68]:


Visualizer(df,'Jordan','new_deaths','vaccination_policy') #Deaths

