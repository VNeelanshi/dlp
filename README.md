# DLP Project Proposal
A quarter long project reproducing a deep learning paper

### Team: Nayan Mehta, KJ Schmidt, Aristana Scourtas, Neelanshi Varia 
### Paper: The Building Blocks of Interpretability

## Paper info:
Title: The Building Blocks of Interpretability <br/>
Link: https://distill.pub/2018/building-blocks/ <br/>
Author: Chris Olah et. al. (Google Brain) <br/>
Author contact: christopherolah.co@gmail.com <br/>
Code repo: https://github.com/distillpub/post--building-blocks

## Literature review:
Most of the Deep Learning papers are mere blackbox in terms of interpretability, decision making, bias and other such aspects. [1] The models give great results but the papers are hardly backed by an explanation of the model’s reasoning behind those results/decisions and mainly talk about the mathematical derivations that led to the architecture. Apart from a few papers which are primarily focused on interpretability, it is usually difficult to find a discussion of interpretability embedded in papers [4]. We were able to find a book on the interpretability of ML which explains importance, methods and other aspects in detail. [10]The following are a few papers in the area: 

Understanding Neural Networks Through Deep Visualization [2]
The paper talks about using two visualization tools to better interpret neural nets, the first one visualizes the activation produced at each layer of a trained Conv net as it takes in image/video and the second allows for visualization of each feature for each layer in the DNN via regularized optimization. 

Using Artificial Intelligence to Augment Human Intelligence [3]
With the advancement in various interactive visualization environments, a lot of dynamic explanations have been emerging. This particular article explains the working behind GANs and interpretation of the intermediary steps for it’s creative applications. They explain this working under a bigger umbrella of questions - what computers are for, and how this relates to intelligence augmentation.

Other similar papers we found which were quite interesting -
- An Evaluation of the Human-Interpretability of Explanation [5]
- “Why Should I Trust You?” Explaining the Predictions of Any Classifier [6]
- Visualising Deep Neural Network Decisions: Prediction Difference Analysis [7]
- Human-in-the-Loop Interpretability Prior [8]
- Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps [9]

Here is the original paper for GoogLeNet (aka Inception v1), which is the model used in our selected paper:

Going Deeper with Convolutions [11]
Describes the implementation of a CNN trained on ImageNet that was at the time (2015) considered state of the art for object detection and classification. “The main hallmark of this architecture is the improved utilization of the computing resources inside the network.” Since this paper, there have been 2 subsequent improved implementations of GoogLeNet: Inception v2 and Inception v3.

## Codebase search: 
- CoLabs used in our paper [In order in which they appear]:
    - https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrChannel.ipynb
    - https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/SemanticDictionary.ipynb
    - https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/ActivationGrid.ipynb
    - https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrSpatial.ipynb
    - https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/NeuronGroups.ipynb
- Lucid -- lib used/written by authors for visualizing neural networks:
    - https://github.com/tensorflow/lucid
- Some tutorials and all of the Colab notebooks created by the authors here:
    - https://github.com/tensorflow/lucid#notebooks 
- Relevant model implementations:
    - GoogLeNet (v1) implementation in Keras (tbd how reliable this is, we should read through the comments)
        - https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14
    - Another implementation of GoogLeNet (v1) (TF)
        - https://github.com/conan7882/GoogLeNet-Inception 
    - Inception v2 and v3 implementations
        - https://github.com/fchollet/deep-learning-models
- Relevant pretrained (presumably on ImageNet) models available:
    - Load pretrained GoogLeNet (v1) and Inception v3 via PyTorch (NOTE: I think we can also use InceptionResNetv2 even though it’s not listed here)
        - https://pytorch.org/docs/stable/torchvision/models.html
    - Alternative PyTorch-compatible GoogLeNet: 
        - https://pypi.org/project/googlenet-pytorch/  
    - Pretrained GoogLeNet (v1) with Caffe
        - https://caffe.berkeleyvision.org/model_zoo.html 
    - Pretrained Inception v2 and v3 in Keras how-to
        - https://gogul.dev/software/flower-recognition-deep-learning 

## Other useful resources: 
- Paper appendix, with info about other channels of GoogleNet<br/>
https://distill.pub/2017/feature-visualization/appendix/
- Twitter walkthrough of the neuron visualizations by the author <br/>
https://twitter.com/ch402/status/927968700384153601
- Book on interpretability in ML <br/>
https://christophm.github.io/interpretable-ml-book/
- Difference between GoogLeNet and Inception v2 and v3 <br/>
https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
- More concise explanation of differences between GoogLeNet and Inception v2 and v3 <br/>
https://datascience.stackexchange.com/questions/15328/what-is-the-difference-between-inception-v2-and-inception-v3 

## Reproducibility review:
We include here the statistically significant features that corresponded to paper reproducibility from (Raff, 2019). 
- *Rigor vs Empirical:* empirical (which is the more reproducible outcome)
- *Readability:* “Excellent.” We feel that we will be able to reproduce the code in a single read.
- *Algorithm Difficulty:* Medium
- *Pseudo Code:* Not present in the article, code contains comments but no explicit pseudocode. Absence of pseudocode was associated with reproducible outcomes.
- *Primary Topic:* Interpretability of Neural Nets; was very easy to identify
- *Hyperparameters Specified:* N/A
- *Compute Needed:* Works easily on Google Colab, does require GPUs
- *Authors Reply:* not checked yet - but they encourage issuing requests on Github, which seems like a sign that they might reply/address the issue. They have also replied to previous reviews.
- *Number Equations:* 0 (in the article), which corresponds to better reproducibility
- *Number Tables:* 4, although we’re defining table loosely here as most of the “tables” are interactive visualizations. However, we think this meets the spirit of the feature.

### Other reproducibility notes:
- Author has some code that is not available: https://twitter.com/ch402/status/928083043654320128?s=20 

### Overall reproducibility:
The paper seems highly reproducible given the criteria set forth by (Raff, 2019). Of note, we would need GPU compute power to run this, but we can do so in Colab. Rewriting the code into Python modules would take some work and create the need for a new compute resource, but this is  manageable. Pretrained versions of GoogLeNet and its subsequent improvements (Inception v2 and v3) are available via PyTorch for ease of use. The paper is very easy to read and understand, and the visualizations provided are well documented and very clear.

## Timeline:
### Week 3 (4/20-4/26):<br/>
Pick project paper<br/>
Project proposal writeup
### Week 4 (4/27-5/3):<br/>
Read the paper completely and get a general overview<br/>
Note down the unclear major concepts and discuss
### Week 5 (5/4-5/10)<br/>
Understand and implement - Making Sense of Hidden Layers
### Week 6 (5/11-5/17)<br/>
Understand and implement - What Does the Network See?
### Week 7 (5/18-5/24)<br/>
Understand and implement - How Are Concepts Assembled?
### Week 8 (5/25-5/31)<br/>
Understand and implement - Making Things Human-Scale
### Week 9 (6/1-6/7)<br/>
Understand and implement -  The Space of Interpretability Interfaces
### Finals Week (6/8-6/13)<br/>
Wrap up<br/>
Conclusion


## References
[1]Yu, R., & Alì, G. (2019). What's Inside the Black Box? AI Challenges for Lawyers and Researchers. Legal Information Management, 19(1), 2-13. doi:10.1017/S1472669619000021 <br/> 
[2]http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf <br/> 
[3] https://distill.pub/2017/aia/ <br/> 
[4] https://arxiv.org/pdf/1710.04806.pdf <br/> 
[5]https://arxiv.org/pdf/1902.00006.pdf <br/> 
[6]https://arxiv.org/pdf/1602.04938.pdf <br/> 
[7]https://arxiv.org/pdf/1702.04595.pdf <br/> 
[8]https://arxiv.org/pdf/1805.11571.pdf <br/> 
[9]https://arxiv.org/pdf/1312.6034.pdf <br/> 
[10]https://christophm.github.io/interpretable-ml-book/ <br/> 
[11] https://arxiv.org/pdf/1409.4842.pdf <br/>
