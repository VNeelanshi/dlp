# Reproducibility Report

## Initial Reproducibility Review
The paper seemed highly reproducible given the criteria set forth by (Raff, 2019). We needed a GPU compute power to run this, but we did so in Colab. Rewriting the code into Python modules could take some work and create the need for a new compute resource, but this is manageable. Pretrained versions of GoogLeNet and its subsequent improvements (Inception v2 and v3) are available via PyTorch for ease of use. The paper is very easy to read and understand, and the visualizations provided are well documented and very clear.

## From Notebooks to Scripts
Our paper’s code consisted of five Google Colab Notebooks. Our first step in reproducing the paper was to make these into python scripts. We were able to do this fairly easily. Our repo contains these scripts that run independently in the scripts folder.

## From Scripts to Modules
Once our scripts were created and running, we set out to use the code to create modules. The modules can be found in src. Our plan was to create module files that contained related functions from the scripts. This meant reorganizing some of the code into groupings that made sense. We created modules from our scripts by placing the code in classes and functions that could be imported elsewhere.

## Visualization
We performed visualization using matplotlib instead of the original lucid library. The visualizations include distribution of channels, spatial distribution of different channels based on the number of layers after calculating the computational graphs. Following are sample images of channels for an image consisting of a cat and a dog. 

![Spritemap 4D Channel](https://github.com/VNeelanshi/dlp/blob/master/scripts/sprite_mixed4d_channel_alpha.jpeg?raw=true)
![Channel distribution](https://github.com/VNeelanshi/dlp/blob/master/data/channel-distribution.png?raw=true)

## Unit tests
We created a set of tests for each of the original notebooks, i.e. each visualization module to check the source files. The test files helped us understand the overall work flow and structure of the code. It also helped us find exact points where the code was breaking. Unlike main.py file, these test scripts test individual module and hence problems in one module will not mess up with other files.
