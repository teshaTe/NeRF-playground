# NeRF-playground

This is a pytorch-based repository with implementation of the Neural Radiance Field (NeRF). All implementations have a detailed description.

### Current implementations:

- Vanilla NeRF

### Requirements

To run the code from this repository the following is essential:
- To run Vanilla NeRF training NVIDIA GPU will be preferable with 8 GB of RAM minimum. However, the code can be run on CPU too.
- Conda environment: needs to be installed and initialized using environment.yml file. All dependencies will be automatically installed. 


### Project description

The repository contains two tools:

- *TrainVanillaNerf_tool.py* - script for training vanilla NeRF model 
- *MeshNerf_tool.py* - script for extracting mesh from the NeRF model. This is a very simple meshing algorithm based on mcubes package.

To run both scripts you will have to open them in editor and tune parameters according to your needs.