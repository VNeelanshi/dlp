While implementing the project we had roughly the following components to take care of the following components
- Environment setup
- Visualization of the building blocks
- Web component
- Version problem
- Model Comparison Spritemaps

## Environment Setup
The documentation and code within the Colab notebooks for package and environment setup was out of date, which was initially quite confusing.  The original code set the Lucid version to 0.0.5 and imported the most recent Tensorflow version, which we thought was intentional, but it turns out the notebooks were just not maintained. Lucid actually required Tensorflow 1.x, and the most up-to-date stable Lucid version available from Pip is 0.3.8, which is significantly updated from 0.0.5. We then had issues with Python versioning and these various packages, and ultimately had to use Python 3.7 even though Lucid is recommended and tested with Python 3.6.

Lucid was not available through Anaconda, which meant we had to do our package management with Pipenv instead of conda. 

## Visualization Problem

Visualizations in the Collaboratory notebooks are all based on Lucid [https://github.com/tensorflow/lucid], a network to visualize neural networks and svelte (web framework) for getting their visualizations to the front end. Svelte [https://svelte.dev/] is not a popular web framework and we couldn’t figure out why the authors chose this. We thought it would make more sense to choose a more common front end framework, like React, that people would be familiar with and would be able to use and build off of easily. We researched Svelte and didn’t see the value in continuing to use it for this project and struggled with the decision of how to show visualizations. We considered React to show them in the browser. However, getting into web development would have been a huge undertaking, so we used Matplotlib for our visualizations instead of displaying them in the browser. 

## Web component

Due to the heavy dependency of the codebase on svelte and lucid, we had to scrape away the browser-dependent components from the original notebooks and instead display the corresponding images with Matplotlib as static representations of the original interactive visualizations. We hope to work on this component in the future with the help of Altair [https://matplotlib.org/mpl-altair/] and gif. 

## Version problem

Lucid is currently being maintained as a research code and not production-ready code. The repository currently does not provide support for Tensorflow 2 and is still limited to Tensorflow 1.x versions. This was initially unclear, as it was only documented by the researchers as of April 15th, 2020. 

## Model Comparison Spritemaps

One of the future scopes of this project was also to reproduce the results using a single model and also allow model comparisons between different models. Although we weren’t able to complete the latter part we did explore how this might be possible to do using this issue from their GitHub repository, https://github.com/tensorflow/lucid/issues/38. This is described as a two-step process where we first obtain a frozen graph definition of the model and next specify the metadata of the model in lucid/modelzoo class. The issue also provided a working notebook with which this can be made possible, although there are still quite a few documented issues on this thread for the notebook listed below.
[https://colab.research.google.com/drive/1PPzeZi5sBN2YRlBmKsdvZPbfYtZI-pHl#scrollTo=EeOS7Fzu4a-p]

