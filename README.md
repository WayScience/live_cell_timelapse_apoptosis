# Live cell timelapse apoptosis analysis

The goal of this project is to develop a pipeline to analyze timelapse images of live cells undergoing apoptosis. The pipeline will be able to detect cells, track them over time, and classify them as either apoptotic or non-apoptotic.
Each dataset contains timelapse images of HeLa cells in a 96-well plate. These cells are treated with varying concentrations of staurosporine, a drug that induces apoptosis.

## Data information

### Doses of staurosporine and replicates

| Staurosporine concentration (nM) | Number of replicates |
|----------------------------------|----------------------|
| 0                                | 3                    |
| 0.61                             | 3                    |
| 1.22                             | 3                    |
| 2.44                             | 3                    |
| 4.88                             | 3                    |
| 9.77                             | 3                    |
| 19.53                            | 3                    |
| 39.06                            | 3                    |
| 78.13                            | 3                    |
| 156.25                           | 3                    |

### Image acquisition

For 4 channel data we acquired at the following wavelengths:

| Channel | Excitation wavelength (nm) | Emission wavelength (nm) |
|---------|-----------------------------|--------------------------|
| Hoecsht | 405 | 447/60 |
| ChromaLive 488 | 488 | 617/73 |
| ChromaLive 488-2 | 488 | 685/40 |
| ChromaLive 561 | 561 | 617/73 |

For 2 channel terminal Annexin V data we acquired at the following wavelengths:

| Channel | Excitation wavelength (nm) | Emission wavelength (nm) |
|---------|-----------------------------|--------------------------|
| Hoecsht | 405 | 447/60 |
| Annexin V | 640 | 685/40 |
