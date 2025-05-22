#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd

# In[2]:


offset_results_path = pathlib.Path("../results")
# get a list of all the offset results
offset_results = list(offset_results_path.glob("*.parquet"))
offset_results.sort()
offset_df = pd.DataFrame(
    pd.concat([pd.read_parquet(offset_result) for offset_result in offset_results])
)
offset_df.to_parquet(
    offset_results_path / "all_offset_results.parquet",
    index=False,
)
