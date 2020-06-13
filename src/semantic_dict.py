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
from utils import googlenet_spritemap

"""
The basic idea of semantic dictionaries is to marry neuron activations to visualizations of those neurons, 
transforming them from abstract vectors to something more meaningful to humans. Semantic dictionaries can 
also be applied to other bases, such as rotated versions of activations space that try to disentangle neurons.
"""

class SemanticDict():
    def __init__(self, model):
        self.model = model

    # NOTE: currently assumes model is GoogLeNet
    def create_semantic_dict(self, layer, img_url):
        img = load(img_url)

        # Compute the activations
        with tf.Graph().as_default(), tf.Session():
            t_input = tf.placeholder(tf.float32, [224, 224, 3])
            T = render.import_model(self.model, t_input, t_input)
            acts = T(layer).eval({t_input: img})[0]

        # Find the most interesting position for our initial view
        max_mag = acts.max(-1)
        max_x = np.argmax(max_mag.max(-1))
        max_y = np.argmax(max_mag[max_x])

        # TODO: make this work for any model, not just GoogLeNet
        # Find appropriate spritemap
        spritemap_n, spritemap_url = googlenet_spritemap(layer)

        imshow(load(spritemap_url))
        print(size(load(spritemap_url)))
        # TODO: create visualization not in Svelte
        # Actually construct the semantic dictionary interface
        # using our *custom component*
        # lucid_svelte.SemanticDict({
        #     "spritemap_url": spritemap_url,
        #     "sprite_size": 110,
        #     "sprite_n_wrap": spritemap_n,
        #     "image_url": _image_url(img),
        #     "activations": [[[{"n": n, "v": float(act_vec[n])} for n in np.argsort(-act_vec)[:4]] for act_vec in act_slice] for act_slice in acts],
        #     "pos" : [max_y, max_x]
        # })
