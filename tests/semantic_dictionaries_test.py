import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url, _display_html

import sys
sys.path.insert(0, '../src')
import unittest
from semantic_dict import SemanticDict

googlenet = models.InceptionV1()
googlenet.load_graphdef()
sd = SemanticDict(googlenet)
sd.create_semantic_dict("mixed4d", "https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")