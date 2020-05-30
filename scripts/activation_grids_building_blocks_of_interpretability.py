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

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.channel_reducer import ChannelReducer
from lucid.misc.io import show, load
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

import src.activation_grid as act


def main():
    model = models.InceptionV1()
    model.load_graphdef()
    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
    # very naive still takes some time to run
    result = act.render_activation_grid_very_naive(img, model, W=48, n_steps=1024)
    result = act.render_activation_grid_less_naive(img, model, W=48, n_steps=1024)

if __name__ == "__main__":
    main()
