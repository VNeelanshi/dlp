from lucid.misc.io import show, load
import sys
sys.path.insert(0, '../src')

from activation_grid import render_activation_grid_less_naive
import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.channel_reducer import ChannelReducer
import sys

from lucid.misc.io import show, load

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")

model = models.InceptionV1()
model.load_graphdef()
_ = render_activation_grid_less_naive(img, model, W=48, n_steps=1024)