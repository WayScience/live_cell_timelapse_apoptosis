# Running scDINO
This module runs scDINO on preprocessed cropped single cell microscopy images.

## Nessessary files and directories
- pyscripts: Directory containing python scripts
    - these scripts are downloaded from the scDINO github repository with the commit hash `3ad7adb05c64a1618c5ce3075a0ae756beff7800`
- only_downstream_analyses.yaml: File containing the configuration for the downstream analyses
- only_downstream_snakefile: File containing the snakemake rules for the downstream analyses

## Running scDINO
To run scDINO, execute the following command:
```bash
source run_scDINO.sh
```
