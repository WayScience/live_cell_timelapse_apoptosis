#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pandas

# In[2]:


output_dir = pathlib.Path("../arrays").resolve()
output_dir.mkdir(exist_ok=True)


# In[3]:


num_of_features = 100
num_of_cells_per_well = 100
num_of_groups = 50
num_of_replicates = 4


def generate_parameter_search_space(
    min,
    max,
    increment,
):
    range = max - min
    num_of_steps = np.floor(range / increment).astype(int)
    return np.linspace(min, max, num_of_steps, dtype=int)


num_of_features = generate_parameter_search_space(10, 3000, 100)
num_of_cells_per_well = generate_parameter_search_space(10, 10000, 1000)
num_of_groups = generate_parameter_search_space(2, 50, 1)
num_of_replicates = generate_parameter_search_space(2, 15, 1)

# get the total number of combinations
total_combinations = (
    len(num_of_features)
    * len(num_of_cells_per_well)
    * len(num_of_groups)
    * len(num_of_replicates)
)
print(f"Total number of combinations: {total_combinations}")


# In[4]:


# write each array to a file
with open(pathlib.Path(f"{output_dir}/num_of_features.txt"), "w") as f:
    for num_of_feature in num_of_features:
        f.write(str(num_of_feature))
        f.write("\n")

with open(pathlib.Path(f"{output_dir}/num_of_cells_per_well.txt"), "w") as f:
    for num_of_cell_per_well in num_of_cells_per_well:
        f.write(str(num_of_cell_per_well))
        f.write("\n")

with open(pathlib.Path(f"{output_dir}/num_of_groups.txt"), "w") as f:
    for num_of_group in num_of_groups:
        f.write(str(num_of_group))
        f.write("\n")

with open(pathlib.Path(f"{output_dir}/num_of_replicates.txt"), "w") as f:
    for num_of_replicate in num_of_replicates:
        f.write(str(num_of_replicate))
        f.write("\n")
