# Processing CP features

## Description
Notbooks:
- [notebooks/0.merge_sc.ipynb](notebooks/0.merge_sc.ipynb):
    - Merging the single cell data with the CP features
- [notebooks/1.annotate_sc.ipynb](notebooks/1.annotate_sc.ipynb):
    - Annotating the single cell data with the CP features
- [notebooks/2.normalize_sc_across_time.ipynb](notebooks/2.normalize_sc_across_time.ipynb):
    - Normalizing the single cell data across time
- [notebooks/2.normalize_sc_within_time.ipynb](notebooks/2.normalize_sc_within_time.ipynb):
    - Normalizing the single cell data within time
- [notebooks/3.feature_select_sc_across_time.ipynb](notebooks/3.feature_select_sc_across_time.ipynb):
    - Feature selection for the single cell data across time
- [notebooks/3.feature_select_sc_within_time.ipynb](notebooks/3.feature_select_sc_within_time.ipynb):
    - Feature selection for the single cell data within time

## When running on HPC:
- Seeing issues with running CytoTable in the same path across multiple nodes and cores :/.
- Typically need to rerun the CytoTable parent script mulitple times to avoid key clashes.
This is not ideal, but it is a workaround.
