# DLP Project Proposal
A quarter long project reproducing a deep learning paper

### Team: Nayan Mehta, KJ Schmidt, Aristana Scourtas, Neelanshi Varia 
### Paper: The Building Blocks of Interpretability

## Paper info:
Title: The Building Blocks of Interpretability
Link: https://distill.pub/2018/building-blocks/
Author: Chris Olah et. al. (Google Brain)
Author contact: christopherolah.co@gmail.com 
Code repo: https://github.com/distillpub/post--building-blocks

## Literature review:
Most of the Deep Learning papers are mere blackbox in terms of interpretability, decision making, bias and other such aspects. [1] The models give great results but the papers are hardly backed by the reasoning behind those results/decisions and mainly talk about the mathematical derivations that led to the architecture. The author has also written a book on the interpretability of ML which explains importance, methods and other aspects in detail. [10] Apart from a few papers which are just about interpretability, it is usually difficult to find that part embedded in papers [4]. Following are a few papers in the area - 

Understanding Neural Networks Through Deep Visualization[2]: The paper talks about using two visualization tools to better interpret neural nets, the first one visualizes the activation produced at each layer of a trained Conv net as it takes in image/video and the second allows for visualization of each feature for each layer in the DNN via regularized optimization. 

Using Artificial Intelligence to Augment Human Intelligence [3]: With the advancement in various interactive visualization environments, a lot of dynamic explanations have been emerging. This particular article explains the working behind GANs and interpretation of the intermediary steps for it’s creative applications. They explain this working under a bigger umbrella of questions - what computers are for, and how this relates to intelligence augmentation.

Other similar papers we found which were quite interesting -
- An Evaluation of the Human-Interpretability of Explanation [5]
- “Why Should I Trust You?” Explaining the Predictions of Any Classifier [6]
- Visualising Deep Neural Network Decisions: Prediction Difference Analysis [7]
- Human-in-the-Loop Interpretability Prior [8]
- Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps [9]

## Codebase search: 
[In order in which they appear in the paper]
- https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrChannel.ipynb 
- https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/SemanticDictionary.ipynb
- https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/ActivationGrid.ipynb
- https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrSpatial.ipynb
- https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/NeuronGroups.ipynb

## Other useful resources: 
- Lucid -- lib used/written by authors for visualizing neural networks:
https://github.com/tensorflow/lucid
- Some tutorials and all of the Colab notebooks created by the authors here:
https://github.com/tensorflow/lucid#notebooks 
- Paper appendix, with info about other channels of GoogleNet
https://distill.pub/2017/feature-visualization/appendix/
- Twitter walkthrough of the neuron visualizations by the author
https://twitter.com/ch402/status/927968700384153601

## Reproducibility review:
We include here the statistically significant features that corresponded to paper reproducibility from (Raff, 2019). 
- Rigor vs Empirical: empirical (which is the more reproducible outcome)
- Readability: “Excellent.” We feel that we will be able to reproduce the code in a single read.
- Algorithm Difficulty: Medium
- Pseudo Code: Not present in the article, code contains comments but no explicit pseudocode. Absence of pseudocode was associated with reproducible outcomes.
- Primary Topic: Interpretability of Neural Nets; was very easy to identify
- Hyperparameters Specified: N/A
- Compute Needed: Works easily on Google Colab
- Authors Reply: not checked yet - but they encourage issuing requests on Github, which seems like a sign that they might reply/address the issue. They have also replied to previous reviews.
- Number Equations: 0 (in the article), which corresponds to better reproducibility
- Number Tables: 4

## Timeline:
- Week 3 (4/20-4/26):
Pick project paper
Project proposal writeup
- Week 4 (4/27-5/3):
Read the paper completely and get a general overview
Note down the unclear major concepts and discuss
- Week 5 (5/4-5/10)
Understand and implement - Making Sense of Hidden Layers
- Week 6 (5/11-5/17)
Understand and implement - What Does the Network See?
- Week 7 (5/18-5/24)
Understand and implement - How Are Concepts Assembled?
- Week 8 (5/25-5/31)
Understand and implement - Making Things Human-Scale
- Week 9 (6/1-6/7)
Understand and implement -  The Space of Interpretability Interfaces
- Finals Week (6/8-6/13)
Wrap up
Conclusion

### Other reproducibility notes: 
We need to make sure that we don’t need to implement GoogleNet (lol) to get this working: https://twitter.com/ch402/status/928083043654320128?s=20 
Update: we do not! But our results will not be as good as those produced by GoogleNet


## References
[1]Yu, R., & Alì, G. (2019). What's Inside the Black Box? AI Challenges for Lawyers and Researchers. Legal Information Management, 19(1), 2-13. doi:10.1017/S1472669619000021
[2]http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf
[3] https://distill.pub/2017/aia/
[4] https://arxiv.org/pdf/1710.04806.pdf
[5]https://arxiv.org/pdf/1902.00006.pdf
[6]https://arxiv.org/pdf/1602.04938.pdf
[7]https://arxiv.org/pdf/1702.04595.pdf
[8]https://arxiv.org/pdf/1805.11571.pdf
[9]https://arxiv.org/pdf/1312.6034.pdf
[10]https://christophm.github.io/interpretable-ml-book/
