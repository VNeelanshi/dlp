# -*- coding: utf-8 -*-
"""Channel Attribution - Building Blocks of Interpretability

Automatically generated by Colaboratory.

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

This colab notebook is part of our **Building Blocks of Intepretability** series exploring how intepretability techniques combine together to explain neural networks. If you haven't already, make sure to look at the [**corresponding paper**](https://distill.pub/2018/building-blocks) as well!

This notebook demonstrates **Channel Attribution**, a technique for exploring how different detectors in the network effected its output.

<br>
<img src="https://storage.googleapis.com/lucid-static/building-blocks/notebook_heroes/channel-attribution.jpeg" width="648"></img>
<br>

This tutorial is based on [**Lucid**](https://github.com/tensorflow/lucid), a network for visualizing neural networks. Lucid is a kind of spiritual successor to DeepDream, but provides flexible abstractions so that it can be used for a wide range of interpretability research.

**Note**: The easiest way to use this tutorial is [as a colab notebook](), which allows you to dive in with no setup. We recommend you enable a free GPU by going:

> **Runtime**   →   **Change runtime type**   →   **Hardware Accelerator: GPU**

Thanks for trying Lucid!

# Install / Import / Load

This code depends on [Lucid](https://github.com/tensorflow/lucid) (our visualization library), and [svelte](https://svelte.technology/) (a web framework). The following cell will install both of them, and dependencies such as TensorFlow. And then import them as appropriate.
"""

# !pip install --quiet lucid==0.0.5
# !npm install -g svelte-cli@2.2.0     # <--- need to install using Node

import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.io import show, load
from lucid.misc.io.reading import read
from lucid.misc.io.showing import _image_url, _display_html
import lucid.scratch.web.svelte as lucid_svelte

model = models.InceptionV1()  # this is GoogLeNet
model.load_graphdef()

"""# Setup (feel free to skip)

**ChannelAttrWidget**

Let's make a little widget for showing all our channels and attribution values.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%html_define_svelte ChannelAttrWidget
# 
# <div class="figure">
#   <div class="channel_list" >
#     {{#each attrsPos as attr}}
#     <div class="entry">
#       <div class="sprite" style="background-image: url({{spritemap_url}}); width: {{sprite_size}}px; height: {{sprite_size}}px; background-position: -{{sprite_size*(attr.n%sprite_n_wrap)}}px -{{sprite_size*Math.floor(attr.n/sprite_n_wrap)}}px;"></div>
#       <div class="value" style="background-color: hsl({{(attr.v > 0)? 210 : 0}}, {{100*Math.abs(attr.v)/1.8}}%, {{100-30*Math.abs(attr.v)/1.8}}%)">{{attr.v}}</div>
#     </div>
#     {{/each}}
#     {{#if attrsPos.length > 5}}
#     <br style="clear:both;">
#     <br style="clear:both;">
#     {{/if}}
#     <div class="gap">...</div>
#     {{#each attrsNeg as attr}}
#     <div class="entry">
#       <div class="sprite" style="background-image: url({{spritemap_url}}); width: {{sprite_size}}px; height: {{sprite_size}}px; background-position: -{{sprite_size*(attr.n%sprite_n_wrap)}}px -{{sprite_size*Math.floor(attr.n/sprite_n_wrap)}}px;"></div>
#       <div class="value" style="background-color: hsl({{(attr.v > 0)? 210 : 0}}, {{100*Math.abs(attr.v)/1.8}}%, {{100-30*Math.abs(attr.v)/1.8}}%)">{{attr.v}}</div>
#     </div>
#     {{/each}}
#   </div>
#   <br style="clear:both">
# </div>
# 
# 
# <style>
#   .entry{
#     float: left;
#     margin-right: 4px;
#   }
#   .gap {
#     float: left;
#     margin: 8px;
#     font-size: 400%;
#   }
# </style>
# 
# <script>
#     
#   function range(n){
#     return Array(n).fill().map((_, i) => i);
#   }
#   
#   export default {
#     data () {
#       return {
#         spritemap_url: "",
#         sprite_size: 110,
#         sprite_n_wrap: 22,
#         attrsPos: [],
#         attrsNeg: [],
#       };
#     },
#     computed: {
#     },
#     helpers: {range}
#   };
# </script>

"""**BarsWidget**

It would also be nice to see the distribution of attribution magnitudes. Let's make another widget for that.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%html_define_svelte BarsWidget
# 
# <div class="figure">
#   <div class="channel_list" >
#     {{#each vals as val}}
#     <div class="bar" style="height: {{15*Math.abs(val)}}px; background-color: hsl({{(val > 0)? 210 : 0}}, {{Math.max(90, 110*Math.abs(val)/1.8)}}%, {{Math.min(80, 100-40*Math.abs(val)/1.8)}}%);">
#     </div>
#     {{/each}}
#   </div>
#   <br style="clear:both">
# </div>
# 
# 
# <style>
#   .channel_list {
#     background-color: #FEFEFE;
#   }
#   .bar {
#     width: 1.5px;
#     height: 10px;
#     display: inline-block;
#   }
# </style>
# 
# <script>
#   
#   export default {
#     data () {
#       return {
#         vals: []
#       };
#     }
#   };
# </script>

"""## **Spritemaps**

In order to show the channels, we need "spritemaps" of channel visualizations.
These visualization spritemaps are large grids of images (such as [this one](https://storage.googleapis.com/lucid-static/building-blocks/sprite_mixed4d_channel.jpeg)) that visualize every channel in a layer.
We provide spritemaps for GoogLeNet because making them takes a few hours of GPU time, but
you can make your own channel spritemaps to explore other models. Check out other notebooks on how to
make your own neuron visualizations.

It's also worth noting that GoogLeNet has unusually semantically meaningful neurons. We don't know why this is -- although it's an active area of research for us. More sophisticated interfaces, such as neuron groups, may work better for networks where meaningful ideas are more entangled or less aligned with the neuron directions.
"""

layer_spritemap_sizes = {
    'mixed3a' : 16,
    'mixed3b' : 21,
    'mixed4a' : 22,
    'mixed4b' : 22,
    'mixed4c' : 22,
    'mixed4d' : 22,
    'mixed4e' : 28,
    'mixed5a' : 28,
  }

def googlenet_spritemap(layer):
  assert layer in layer_spritemap_sizes
  size = layer_spritemap_sizes[layer]
  url = "https://storage.googleapis.com/lucid-static/building-blocks/googlenet_spritemaps/sprite_%s_channel_alpha.jpeg" % layer
  return size, url

"""**Attribution Code**"""

def score_f(logit, name):
  if name is None:
    return 0
  elif name == "logsumexp":
    base = tf.reduce_max(logit)
    return base + tf.log(tf.reduce_sum(tf.exp(logit-base)))
  elif name in model.labels:
    return logit[model.labels.index(name)]
  else:
    raise RuntimeError("Unsupported")

def channel_attr_simple(img, layer, class1, class2, n_show=4):

  # Set up a graph for doing attribution...
  with tf.Graph().as_default(), tf.Session() as sess:
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    
    # Compute activations
    acts = T(layer).eval()
    
    # Compute gradient
    logit = T("softmax2_pre_activation")[0]
    score = score_f(logit, class1) - score_f(logit, class2)
    t_grad = tf.gradients([score], [T(layer)])[0]
    grad = t_grad.eval()
    
    # Let's do a very simple linear approximation attribution.
    # That is, we say the attribution of y to x is 
    # the rate at which x changes y times the value of x.
    attr = (grad*acts)[0]
    
    # Then we reduce down to channels.
    channel_attr = attr.sum(0).sum(0)

  # Now we just need to present the results.
  
  # Get spritemaps
  
  
  spritemap_n, spritemap_url = googlenet_spritemap(layer)
  
  # Let's show the distribution of attributions
  print("Distribution of attribution accross channels:")
  print("")
  lucid_svelte.BarsWidget({"vals" : [float(v) for v in np.sort(channel_attr)[::-1]]})

  # Let's pick the most extreme channels to show
  ns_pos = list(np.argsort(-channel_attr)[:n_show])
  ns_neg = list(np.argsort(channel_attr)[:n_show][::-1])
  
  # ...  and show them with ChannelAttrWidget
  print("")
  print("Top", n_show, "channels in each direction:")
  print("")
  lucid_svelte.ChannelAttrWidget({
    "spritemap_url": spritemap_url,
    "sprite_size": 110,
    "sprite_n_wrap": spritemap_n,
    "attrsPos": [{"n": n, "v": str(float(channel_attr[n]))[:5]} for n in ns_pos],
    "attrsNeg": [{"n": n, "v": str(float(channel_attr[n]))[:5]} for n in ns_neg] 
  })

"""# Channel attributions from article teaser"""

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
channel_attr_simple(img, "mixed4d", "Labrador retriever", "tiger cat", n_show=3)

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/flowers.png")
channel_attr_simple(img, "mixed4d", "vase", "lemon", n_show=3)

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/sunglasses_tux.png")
channel_attr_simple(img, "mixed4d", "bow tie", "sunglasses", n_show=3)

"""# Bigger channel attribution!!!"""

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
channel_attr_simple(img, "mixed4d", "Labrador retriever", "tiger cat", n_show=30)

"""# Channel Attribution - Path Integrated"""

def channel_attr_path(img, layer, class1, class2, n_show=4, stochastic_path=False, N = 100):

  # Set up a graph for doing attribution...
  with tf.Graph().as_default(), tf.Session() as sess:
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)
    
    # Compute activations
    acts = T(layer).eval()
    
    # Compute gradient
    logit = T("softmax2_pre_activation")[0]
    score = score_f(logit, class1) - score_f(logit, class2)
    t_grad = tf.gradients([score], [T(layer)])[0]

    
    # Inegrate on a path from acts=0 to acts=acts
    attr = np.zeros(acts.shape[1:])
    for n in range(N):
      acts_ = acts * float(n) / N
      if stochastic_path:
        acts_ *= (np.random.uniform(0, 1, [528])+np.random.uniform(0, 1, [528]))/1.5
      grad = t_grad.eval({T(layer): acts_})
      attr += 1.0 / N * (grad*acts)[0]
    
    # Then we reduce down to channels.
    channel_attr = attr.sum(0).sum(0)

  # Now we just need to present the results.
  
  # Get spritemaps
  
  
  spritemap_n, spritemap_url = googlenet_spritemap(layer)
  
  # Let's show the distribution of attributions
  print("Distribution of attribution accross channels:")
  print("")
  lucid_svelte.BarsWidget({"vals" : [float(v) for v in np.sort(channel_attr)[::-1]]})

  # Let's pick the most extreme channels to show
  ns_pos = list(np.argsort(-channel_attr)[:n_show])
  ns_neg = list(np.argsort(channel_attr)[:n_show][::-1])
  
  # ...  and show them with ChannelAttrWidget
  print("")
  print("Top", n_show, "channels in each direction:")
  print("")
  lucid_svelte.ChannelAttrWidget({
    "spritemap_url": spritemap_url,
    "sprite_size": 110,
    "sprite_n_wrap": spritemap_n,
    "attrsPos": [{"n": n, "v": str(float(channel_attr[n]))[:5]} for n in ns_pos],
    "attrsNeg": [{"n": n, "v": str(float(channel_attr[n]))[:5]} for n in ns_neg] 
  })

def compare_attr_methods(img, class1, class2):
  
  _display_html("<h2>Linear Attribution</h2>")
  channel_attr_simple(img, "mixed4d", class1, class2, n_show=10)

  _display_html("<br><br><h2>Path Integrated Attribution</h2>")
  channel_attr_path(img, "mixed4d", class1, class2, n_show=10)
  
  _display_html("<br><br><h2>Stochastic Path Integrated Attribution</h2>")
  channel_attr_path(img, "mixed4d", class1, class2, n_show=10, stochastic_path=True)

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")

compare_attr_methods(img, "Labrador retriever", "tiger cat")

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/flowers.png")

compare_attr_methods(img, "vase", "lemon")

img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/pig.jpeg")

compare_attr_methods(img, "hog", "dalmatian")

