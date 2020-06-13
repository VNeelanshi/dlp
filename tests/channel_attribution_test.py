import lucid.modelzoo.vision_models as models
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url, _display_html

import sys
sys.path.insert(0, '../src')
# import unittest
from attribution import ChannelAttribution

model = models.InceptionV1()  # this is GoogLeNet
model.load_graphdef()

def compare_attr_methods(attr, img, class1, class2):
    _display_html("<h2>Linear Attribution</h2>")
    attr.channel_attr(img, "mixed4d", class1, class2, mode="simple", n_show=10)

    _display_html("<br><br><h2>Path Integrated Attribution</h2>")
    attr.channel_attr(img, "mixed4d", class1, class2, mode="path", n_show=10)

    _display_html("<br><br><h2>Stochastic Path Integrated Attribution</h2>")
    attr.channel_attr(img, "mixed4d", class1, class2, mode="path", n_show=10, stochastic_path=True)

attr = ChannelAttribution(model)
"""# Channel attributions from article teaser"""
img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
attr.channel_attr(img, "mixed4d", "Labrador retriever", "tiger cat", mode="simple", n_show=3)