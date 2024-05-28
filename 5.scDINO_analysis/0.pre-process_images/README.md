# Preprocessing images for scDINO
This module saves multichannel single-cell cropped images in `.tiff` format.
This allows for the scDINO module to be run correctly.
The scDINO module runs on preprocessed cropped single cell microscopy images.
From these images, the module extracts the single-cell representations.

## Runnninng the module
To run the module, execute the following command:
```bash
run_image_preprocess.sh
```

## Run time
The module takes approximately hours to run for 210,217 grayscale microscopy images.

This was run on a system with the following specifications:
- Processor: AMD Ryzen 9 5900X (24) @ 3.700GHz
- Memory: 128GB DDR4
- GPU: NVIDIA GeForce RTX 3090 TI
- OS: Pop!_OS 22.04 LTS x86_64
- Kernel: 6.4.6-76060406-generic
- Shell: bash 5.1.16
