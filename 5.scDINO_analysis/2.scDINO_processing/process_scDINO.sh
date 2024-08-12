#!/bin/bash
# This script runs the scDINO processing notebooks

# activate the correct env
conda activate scDINO_env

# change dir to the notebook folder
cd notebooks

# run the notebooks
papermill 0.scDINO_processing_individual_channels.ipynb 0.scDINO_processing_individual_channels.ipynb
papermill 1.calculate_umap_embeddings.ipynb 1.calculate_umap_embeddings.ipynb

# deactivate the env and activate the R env
conda deactivate
conda activate R_env
# run the R notebook
papermill 2.scDINO_cls_token_visualization.ipynb 2.scDINO_cls_token_visualization.ipynb

# deactivate the R env and activate the scDINO env
conda deactivate
conda activate scDINO_env
# run the python notebook
papermill 3.scDINO_cls_timelapse_vizualization.ipynb 3.scDINO_cls_timelapse_vizualization.ipynb

# change dir back to the original folder
cd ..

# convert the ran notebooks to scripts
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

# deactivate the env
conda deactivate

# Complete
echo "scDINO processing complete"
