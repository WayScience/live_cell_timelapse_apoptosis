#!/usr/bin/env python
# coding: utf-8

# This notebook fixes issues with data paths by replacing spaces with underscores

# In[1]:


import pathlib

# In[2]:


# set data path
data_path = pathlib.Path("../../data").resolve(strict=True)


# In[3]:


# get all directories in data path
directories = [x for x in data_path.iterdir() if x.is_dir()]
directories


# In[4]:


# rename the directories to replace spaces with underscores
for directory in directories:
    new_name = directory.name.replace(" ", "_")
    directory.rename(directory.parent / new_name)


# In[5]:


# get all directories in data path
directories = [x for x in data_path.iterdir() if x.is_dir()]
directories
