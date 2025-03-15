#!/usr/bin/env python
# coding: utf-8

# # Perform feature selection on normalized data

# ## Import libraries

# In[1]:


import argparse
import gc
import pathlib

import numpy as np
import pandas as pd
import psutil
from memory_profiler import profile
from pycytominer import feature_select
from pycytominer.cyto_utils import output

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    print("Running as script")
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_of_features", type=int, default=1000)
    argparser.add_argument("--num_of_cells_per_well", type=int, default=100)
    argparser.add_argument("--num_of_groups", type=int, default=50)
    argparser.add_argument("--num_of_replicates", type=int, default=4)
    args = argparser.parse_args()

    num_of_features = args.num_of_features
    num_of_cells_per_well = args.num_of_cells_per_well
    num_of_groups = args.num_of_groups
    num_of_replicates = args.num_of_replicates
else:
    num_of_features = 100
    num_of_cells_per_well = 100
    num_of_groups = 50
    num_of_replicates = 4


# In[3]:


prior_process = psutil.Process()
prior_mem_info = prior_process.memory_info()


# ## Perform feature selection

# In[4]:


# define operations to be performed on the data
# list of operations for feature select function to use on input profile
feature_select_ops = [
    "variance_threshold",
    "blocklist",
    "drop_na_columns",
    "correlation_threshold",
]


# In[5]:


# generate a profile for input data for fs
def generate_toy_data(
    num_of_features: int = 1000,
    num_of_cells_per_well: int = 100,
    num_of_groups: int = 50,
    num_of_replicates: int = 4,
    seed: int = 0,
):
    np.random.seed(seed)
    num_of_rows_total = num_of_groups * num_of_replicates
    output_dict = {
        "Metadata_Well": [],
    }
    for group in range(num_of_groups):
        for replicate in range(num_of_replicates):
            well = f"{group}_{replicate}"
            output_dict["Metadata_Well"].extend([well] * num_of_cells_per_well)

    for feature in range(num_of_features):
        feature_name = f"feature_{feature}"
        output_dict[feature_name] = np.random.normal(
            0, 1, num_of_rows_total * num_of_cells_per_well
        )

    df = pd.DataFrame(output_dict)
    return df


df = generate_toy_data(
    num_of_features=num_of_features,
    num_of_cells_per_well=num_of_cells_per_well,
    num_of_groups=num_of_groups,
    num_of_replicates=num_of_replicates,
    seed=0,
)


# In[ ]:


metadata_cols = ["Metadata_Well"]
feature_cols = [x for x in df.columns if x not in metadata_cols]
feature_select_df = feature_select(
    df,
    operation=feature_select_ops,
    features=feature_cols,
)

num_of_features_retained = feature_select_df.shape[1]
percent_of_features_retained = num_of_features_retained / df.shape[1] * 100
print(f"Initial shape: {df.shape}, Final shape: {feature_select_df.shape}")
print(f"Number of features retained: {num_of_features_retained}")
print(f"Percent of features retained: {percent_of_features_retained:.2f}%")
del df
del feature_select_df


# In[7]:


post_process = psutil.Process()
post_mem_info = post_process.memory_info()

print(f"RSS: {(post_mem_info.rss - prior_mem_info.rss) / (1024 ** 2):.2f} MB")
print(f"VMS: {(post_mem_info.vms - prior_mem_info.vms) / (1024 ** 2):.2f} MB")
print(f"Shared: {(post_mem_info.shared - prior_mem_info.shared) / (1024 ** 2):.2f} MB")
print(f"Text: {(post_mem_info.text - prior_mem_info.text) / (1024 ** 2):.2f} MB")
print(f"Lib: {(post_mem_info.lib - prior_mem_info.lib) / (1024 ** 2):.2f} MB")
print(f"Data: {(post_mem_info.data - prior_mem_info.data) / (1024 ** 2):.2f} MB")
print(f"Dirty: {(post_mem_info.dirty - prior_mem_info.dirty) / (1024 ** 2):.2f} MB")
print(f"Total: {(post_mem_info.rss - prior_mem_info.rss) / (1024 ** 2):.2f} MB")

output_dict = {
    "num_of_features": num_of_features,
    "num_of_cells_per_well": num_of_cells_per_well,
    "num_of_groups": num_of_groups,
    "num_of_replicates": num_of_replicates,
    "num_of_features_retained": num_of_features_retained,
    "percent_of_features_retained": percent_of_features_retained,
    "rss_MB": (post_mem_info.rss - prior_mem_info.rss) / (1024**2),
    "vms_MB": (post_mem_info.vms - prior_mem_info.vms) / (1024**2),
    "shared_MB": (post_mem_info.shared - prior_mem_info.shared) / (1024**2),
    "text_MB": (post_mem_info.text - prior_mem_info.text) / (1024**2),
    "lib_MB": (post_mem_info.lib - prior_mem_info.lib) / (1024**2),
    "data_MB": (post_mem_info.data - prior_mem_info.data) / (1024**2),
    "dirty_MB": (post_mem_info.dirty - prior_mem_info.dirty) / (1024**2),
    "total_MB": (post_mem_info.rss - prior_mem_info.rss) / (1024**2),
}
output_df = pd.DataFrame(output_dict, index=[0])
output_df


# In[8]:


# write the output to a file
output_file_path = pathlib.Path(
    f"../results_of_memory_profiling/{num_of_features}_features_{num_of_cells_per_well}_cells_per_well_{num_of_groups}_groups_{num_of_replicates}_replicates.parquet"
).resolve()
output_file_path.parent.mkdir(exist_ok=True, parents=True)
output_df.to_parquet(output_file_path)
