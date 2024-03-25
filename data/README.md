# Data sets and directories
Each directory is a separate dataset.
The directory structure is as follows:


## 24hr 4 channel data
These datasets contain 24 hour timelapse data with 4 channels at 10 minute intervals.

#### Metadata: [metadata_24h.csv](metadata_24h.csv)
#### Raw data: [20230920ChromaLiveTL_24hr4ch](20230920ChromaLiveTL_24hr4ch)
#### Maximal intensity projections: [20230920ChromaLiveTL_24hr4ch_MaxIP](20230920ChromaLiveTL_24hr4ch_MaxIP)

## 6hr 4 channel data
These datasets contain 6 hour timelapse data with 4 channels at 30 minute intervals.
#### Metadata: [metadata_6hr_4ch.csv](metadata_6hr_4ch.csv)
#### Raw data: [20231004ChromaLive6hr_4ch](20231004ChromaLive6hr_4ch)
#### Maximal intensity projections: [20231004ChromaLive6hr_4ch_MaxIP](20231004ChromaLive6hr_4ch_MaxIP)

## 6hr 4 channel data + Endpoint data with Annexin V Data
These datasets contain 6 hour timelapse data with 4 channels at 30 minute intervals.
These datasets also contain endpoint data with Annexin V staining + DAPI.
#### Metadata:
[metadata_6hr_4ch.csv](metadata_6hr_4ch.csv)
#### Raw data: [20231017ChromaLive_6hr_4ch](20231017ChromaLive_6hr_4ch)
#### Maximal intensity projections: [20231017ChromaLive_6hr_4ch_MaxIP](20231017ChromaLive_6hr_4ch_MaxIP)

## Endpoint raw data:
#### Metadata: [metadata_AnnexinV_2ch.csv](metadata_AnnexinV_2ch.csv)
#### Raw data: [20231017ChromaLive_endpoint_w_AnnexinV_2ch](20231017ChromaLive_endpoint_w_AnnexinV_2ch)
#### Endpoint maximal intensity projections: [20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP](20231017ChromaLive_endpoint_w_AnnexinV_2ch_MaxIP)

# File naming convention
Example file name: `C-02_F0001_T0001_Z0001_C01.tif`
- `C-02`: Well C02 (96-well plate)
- `F0001`: Field of view 1
- `T0001`: Timepoint 1
- `Z0001`: Z-slice 1
    - Alternative: Z0001 is the maximal intensity projection
- `C01`: Channel 1
