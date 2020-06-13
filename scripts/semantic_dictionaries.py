# -*- coding: utf-8 -*-
"""Semantic Dictionaries - Building Blocks of Interpretability

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/SemanticDictionary.ipynb

##### Copyright 2018 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The basic idea of semantic dictionaries is to marry neuron activations to visualizations of those neurons, 
transforming them from abstract vectors to something more meaningful to humans. Semantic dictionaries can also be 
applied to other bases, such as rotated versions of activations space that try to disentangle neurons.

This code depends on [Lucid](https://github.com/tensorflow/lucid) (our visualization library), and 
[svelte](https://svelte.technology/) (a web framework). The following cell will install both of them, and 
dependancies such as TensorFlow. And then import them as appropriate.
"""

import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url
# import lucid.scratch.web.svelte as lucid_svelte

import sys
sys.path.insert(0, '../src')
import unittest
from semantic_dict import SemanticDict


"""# Semantic Dictionary Code

## **Defining the interface**

First, we define our "semantic dictionary" interface as a [svelte component](https://svelte.technology/). This makes it easy to manage state, like which position we're looking at.
"""

"""## **Spritemaps**

In order to use the semantic dictionaries, we need "spritemaps" of channel visualizations.
These visualization spritemaps are large grids of images (such as [this one](https://storage.googleapis.com/lucid-static/building-blocks/googlenet_spritemaps/sprite_mixed4d_channel.jpeg)) that visualize every channel in a layer.
We provide spritemaps for GoogLeNet because making them takes a few hours of GPU time, but
you can make your own channel spritemaps to explore other models. [Check out other notebooks](https://github.com/tensorflow/lucid#notebooks) on how to
make your own neuron visualizations.

It's also worth noting that GoogLeNet has unusually semantically meaningful neurons. We don't know why this is -- although it's an active area of research for us. More sophisticated interfaces, such as neuron groups, may work better for networks where meaningful ideas are more entangled or less aligned with the neuron directions.
"""

"""## **User facing constructor**

Now we'll create a convenient API for creating semantic dictionary visualizations. It will compute the network activations for an image, grab an appropriate spritemap, and render the interface.
"""


"""# Now let's make some semantic dictionaries!"""
def main():
    googlenet = models.InceptionV1()
    googlenet.load_graphdef()
    sd = SemanticDict(googlenet)
    sd.create_semantic_dict("mixed4d", "https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
    sd.create_semantic_dict("mixed4d", "https://storage.googleapis.com/lucid-static/building-blocks/examples/flowers.png")


if __name__ == "__main__":
    main()
