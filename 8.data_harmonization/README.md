# Morphology Data Harmonization
This module harmonizes the moprhology feature space from CellProfiller features and scDINO features.

## Running the notebooks
To Harmonize the morphology feature space run the command:
```bash
source run_data_harmonization.sh
```



Metdata column descriptions:
| Metadata column name | Description |
|----------------------|-------------|
| Metadata_plate | The plate identifier for the experiment. |
| Metadata_Well | The well identifier for the experiment. |
| Metadata_number_of_singlecells | The number of single cells in the image. |
| Metadata_compound | The compound identifier for the experiment. |
| Metadata_dose | The dose of the compound used in the experiment. |
| Metadata_control | A flag indicating if the sample is a control. |
| Metadata_ImageNumber | The image number for the experiment. |
| Metadata_FOV | The field of view identifier for the experiment. |
| Metadata_Time | The time point for the experiment. |
| Metadata_Cells_Number_Object_Number | The number of cells in the image. |
| Metadata_Cytoplasm_Parent_Cells | The parent cells for the cytoplasm. |
| Metadata_Cytoplasm_Parent_Nuclei | The parent nuclei for the cytoplasm. |
| Metadata_ImageNumber_1 | The first image number for the experiment. |
| Metadata_ImageNumber_2 | The second image number for the experiment. |
| Metadata_ImageNumber_3 | The third image number for the experiment. |
| Metadata_Nuclei_Number_Object_Number | The number of nuclei in the image. |
| Metadata_Image_FileName_CL_488_1 | The file name for the CL-488 image (first). |
| Metadata_Image_FileName_CL_488_2 | The file name for the CL-488 image (second). |
| Metadata_Image_FileName_CL_561 | The file name for the CL-561 image. |
| Metadata_Image_FileName_DNA | The file name for the DNA image. |
| Metadata_Image_PathName_CL_488_2 | The path name for the CL-488 image (second). |
| Metadata_Image_PathName_CL_561 | The path name for the CL-561 image. |
| Metadata_Nuclei_Location_Center_X | The x-coordinate of the nucleus location. |
| Metadata_Nuclei_Location_Center_Y | The y-coordinate of the nucleus location. |
| Metadata_coordinates_x | The x-coordinate of the object. |
| Metadata_track_id | The track identifier for the object. |
| Metadata_t | The time point for the object. |
| Metadata_y | The y-coordinate of the object. |
| Metadata_x | The x-coordinate of the object. |
| Metadata_id | The identifier for the object. |
| Metadata_parent_track_id | The track identifier for the parent object. |
| Metadata_parent_id | The identifier for the parent object. |
| Metadata_coordinates_y | The y-coordinate of the object. |
| Metadata_distance | The distance for the object. |
| Metadata_original_index_x | The original x-index for the object. |
| Metadata_image_path | The path for the image. |
| Metadata_original_index_y | The original y-index for the object. |
