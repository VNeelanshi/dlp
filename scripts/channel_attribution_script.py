# -*- coding: utf-8 -*-
"""Channel Attribution - Building Blocks of Interpretability

Automatically generated by Colaboratory, then amended by Aristana Scourtas.

Original file is located at
    https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrChannel.ipynb

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

"""# Channel Attribution -- Building Blocks of Interpretability

This script is part of the **Building Blocks of Intepretability** series exploring how intepretability techniques combine together to explain neural networks. If you haven't already, make sure to look at the [**corresponding paper**](https://distill.pub/2018/building-blocks) as well!

This demonstrates **Channel Attribution**, a technique for exploring how different detectors in the network effected its output.
"""
import lucid.modelzoo.vision_models as models
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url, _display_html
# import lucid.scratch.web.svelte as lucid_svelte  ## not using svelte just yet

#from src.attribution import ChannelAttribution
import sys
sys.path.insert(0, '../src')
import unittest
from attribution import ChannelAttribution

"""## **Spritemaps**

In order to show the channels, we need "spritemaps" of channel visualizations.
These visualization spritemaps are large grids of images (such as [this one](https://storage.googleapis.com/lucid-static/building-blocks/sprite_mixed4d_channel.jpeg)) that visualize every channel in a layer.
We provide spritemaps for GoogLeNet because making them takes a few hours of GPU time, but
you can make your own channel spritemaps to explore other models. Check out other notebooks on how to
make your own neuron visualizations.

It's also worth noting that GoogLeNet has unusually semantically meaningful neurons. We don't know why this is -- although it's an active area of research for us. More sophisticated interfaces, such as neuron groups, may work better for networks where meaningful ideas are more entangled or less aligned with the neuron directions.
"""
"""**Attribution Code**"""

model = models.InceptionV1()  # this is GoogLeNet
model.load_graphdef()

def compare_attr_methods(attr, img, class1, class2):
    _display_html("<h2>Linear Attribution</h2>")
    attr.channel_attr(img, "mixed4d", class1, class2, mode="simple", n_show=10)

    _display_html("<br><br><h2>Path Integrated Attribution</h2>")
    attr.channel_attr(img, "mixed4d", class1, class2, mode="path", n_show=10)

    _display_html("<br><br><h2>Stochastic Path Integrated Attribution</h2>")
    attr.channel_attr(img, "mixed4d", class1, class2, mode="path", n_show=10, stochastic_path=True)


# TODO: replace all lucid_svelte calls with alt visualization
def main():
    attr = ChannelAttribution(model)
    """# Channel attributions from article teaser"""
    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
    attr.channel_attr(img, "mixed4d", "Labrador retriever", "tiger cat", mode="simple", n_show=3)

    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/flowers.png")
    attr.channel_attr(img, "mixed4d", "vase", "lemon", mode="simple", n_show=3)

    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/sunglasses_tux.png")
    attr.channel_attr(img, "mixed4d", "bow tie", "sunglasses", mode="simple", n_show=3)

    """# Bigger channel attribution!!!"""
    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
    attr.channel_attr(img, "mixed4d", "Labrador retriever", "tiger cat", mode="simple", n_show=30)

    """# Channel Attribution - Path Integrated"""
    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
    compare_attr_methods(attr, img, "Labrador retriever", "tiger cat")

    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/flowers.png")
    compare_attr_methods(attr, img, "vase", "lemon")

    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/pig.jpeg")
    compare_attr_methods(attr, img, "hog", "dalmatian")


if __name__ == "__main__":
    main()
