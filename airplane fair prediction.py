#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[3]:


traindata=pd.read_excel(r"datatrain.xlsx")


# In[4]:


pd.set_option('display.max_columns',None)


# In[5]:


traindata.head()


# In[6]:


traindata.info()


# In[7]:


traindata["Duration"].value_counts()


# In[9]:


traindata.shape


# In[10]:


traindata.dropna(inplace = True)


# In[11]:


traindata.isnull().sum()


# In[12]:


traindata["Journey_day"]= pd.to_datetime(traindata.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[13]:


traindata["Journey_month"]= pd.to_datetime(traindata["Date_of_Journey"], format= "%d/%m/%Y").dt.month


# In[14]:


traindata.head()


# In[15]:


traindata.drop(["Date_of_Journey"],axis= 1, inplace= True)


# In[16]:


traindata["Dep_hour"]= pd.to_datetime(traindata["Dep_Time"]).dt.hour

traindata["Dep_min"]= pd.to_datetime(traindata["Dep_Time"]).dt.minute

traindata.drop(["Dep_Time"], axis= 1, inplace= True)


# In[17]:


traindata.head()


# In[18]:


traindata["Arrival_hour"]= pd.to_datetime(traindata["Arrival_Time"]).dt.hour

traindata["Arrival_min"]= pd.to_datetime(traindata["Arrival_Time"]).dt.minute

traindata.drop(["Arrival_Time"], axis= 1, inplace= True)


# In[19]:


traindata.head()


# In[29]:


traindata["Airline"].value_counts()


# In[31]:


sns.catplot(y = "Price", x = "Airline", data = traindata.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[32]:


Airline =traindata[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[36]:


traindata["Source"].value_counts()


# In[40]:


sns.catplot(y = "Price", x ="Source", data = traindata.sort_values("Price", ascending= False),kind="boxen", height=4, aspect=3)
plt.show()


# In[44]:


Source = traindata[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[45]:


traindata["Destination"].value_counts()


# In[46]:


Destination= traindata[["Destination"]]

Destination= pd.get_dummies(Destination, drop_first= True)

Destination.head()


# In[47]:


traindata["Route"]


# In[49]:


traindata.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[50]:


traindata["Total_Stops"].value_counts()

traindata.replace({"non-stop":0,"1 stop":1,"2 stop":2,"3 stop":3,"4 stop":4}, inplace=True)
# In[51]:


traindata.head()


# In[52]:


tarindata= pd.concat([traindata,Airline,Source,Destination], axis=1 )


# In[53]:


tarindata.head()


# In[54]:


traindata.drop(["Airline","Source","Destination"], axis=1,inplace=True)


# In[55]:


traindata.head()


# In[56]:


traindata.shape


# In[58]:


test_data= pd.read_excel("testset.xlsx")


# In[59]:


test_data.head()


# In[60]:


print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour
            
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[72]:


data_test.head()


# In[78]:


traindata.shape


# In[81]:


traindata.columns


# In[ ]:





# In[84]:


y = traindata.iloc[:, 1]
y.head()


# In[86]:


plt.figure(figsize = (8,8))
sns.heatmap(traindata.corr(), annot = True, cmap = "RdYlGn")

plt.show()

