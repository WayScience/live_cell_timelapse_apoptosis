#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd

# In[2]:


well_fov_dir = pathlib.Path(
    "../../2.cellprofiler_ic_processing/illum_directory/timelapse"
).resolve(strict=True)
well_fov_dirs = list(well_fov_dir.glob("*"))
well_fov_dirs = [d for d in well_fov_dirs if d.is_dir()]
well_fov_dirs = sorted(well_fov_dirs)
well_fov_dirs = [x.name.split("_MaxIP_")[1] for x in well_fov_dirs]


# In[3]:


# write the well_fov_dirs to a file for a bash script to read
well_fov_dirs = pd.Series(well_fov_dirs)
pathlib.Path("../well_fov_loading/").mkdir(parents=True, exist_ok=True)
well_fov_dirs.to_csv("../well_fov_loading/well_fov_dirs.csv", index=False, header=False)
