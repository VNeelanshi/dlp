# Reproducibility Report

## From Notebooks to Scripts
Our paper’s code consisted of different Google Colab Notebooks. Our first step in reproducing the paper was to make these into python scripts. We were able to do this fairly easily. Our repo contains these scripts that run independently in the scripts folder.

## From Scripts to Modules
One our scripts were created and running, we set out to use the code to create modules. The modules can be found in src. Our plan was to create module files that contained related functions from the scripts. This meant reorganizing some of the code into groupings that made sense. We created modules from our scripts by placing the code in classes and functions that could be imported elsewhere. We planned to import and call our modules in main.py.

## New Work Flow
In main.py we needed to understand what functions to call in the correct order for everything to run smoothly.

## Visualization Problem
The original paper uses Svelte as the framework for getting their visualizations to the front end. We researched Svelte and didn’t see the value in continuing to use it for this project. We struggled with the decision of how to show visualizations. We considered React to show them in the browser. However, with the limited time in the quarter, this would be a huge undertaking. We also considered Matplotlib as a starting point for visualizations, and a few of our scripts and modules do make use of this for their visualizations.