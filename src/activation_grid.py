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

'''This module contains code to render the activation grid'''


class ActivationGrid():
    def __init__(self, model):
        self.model = model

    def render_activation_grid_very_naive(self, img, layer="mixed4d", W=42, n_steps=256):

        # Get the activations
        with tf.Graph().as_default(), tf.Session() as sess:
            t_input = tf.placeholder("float32", [None, None, None, 3])
            T = render.import_model(self.model, t_input, t_input)
            acts = T(layer).eval({t_input: img[None]})[0]
        acts_flat = acts.reshape([-1] + [acts.shape[2]])

        # Render an image for each activation vector
        def param_f(): return param.image(W, batch=acts_flat.shape[0])
        obj = objectives.Objective.sum(
            [objectives.direction(layer, v, batch=n)
             for n, v in enumerate(acts_flat)
             ])
        thresholds = (n_steps//2, n_steps)
        vis_imgs = render.render_vis(self.model, obj, param_f, thresholds=thresholds)[-1]

        # Combine the images and display the resulting grid
        vis_imgs_ = vis_imgs.reshape(list(acts.shape[:2]) + [W, W, 3])
        vis_imgs_cropped = vis_imgs_[:, :, 2:-2, 2:-2, :]
        show(np.hstack(np.hstack(vis_imgs_cropped)))
        return vis_imgs_cropped

    def render_activation_grid_less_naive(self, img, layer="mixed4d", W=42, n_groups=6, subsample_factor=1, n_steps=256):
        # Get the activations
        with tf.Graph().as_default(), tf.Session() as sess:
            t_input = tf.placeholder("float32", [None, None, None, 3])
            T = render.import_model(self.model, t_input, t_input)
            acts = T(layer).eval({t_input: img[None]})[0]
        acts_flat = acts.reshape([-1] + [acts.shape[2]])
        N = acts_flat.shape[0]

        # The trick to avoiding "decoherence" is to recognize images that are
        # for similar activation vectors and
        if n_groups > 0:
            reducer = ChannelReducer(n_groups, "NMF")
            groups = reducer.fit_transform(acts_flat)
            groups /= groups.max(0)
        else:
            groups = np.zeros([])

        print(groups.shape)

        # The key trick to increasing memory efficiency is random sampling.
        # Even though we're visualizing lots of images, we only run a small
        # subset through the network at once. In order to do this, we'll need
        # to hold tensors in a tensorflow graph around the visualization process.
        with tf.Graph().as_default() as graph, tf.Session() as sess:
            # Using the groups, create a paramaterization of images that
            # partly shares paramters between the images for similar activation
            # vectors. Each one still has a full set of unique parameters, and could
            # optimize to any image. We're just making it easier to find solutions
            # where things are the same.
            group_imgs_raw = param.fft_image([n_groups, W, W, 3])
            unique_imgs_raw = param.fft_image([N, W, W, 3])
            opt_imgs = param.to_valid_rgb(tf.stack([
                0.7*unique_imgs_raw[i] +
                0.5*sum(groups[i, j] * group_imgs_raw[j] for j in range(n_groups))
                for i in range(N)]),
                decorrelate=True)

            # Construct a random batch to optimize this step
            batch_size = 64
            rand_inds = tf.random_uniform([batch_size], 0, N, dtype=tf.int32)
            pres_imgs = tf.gather(opt_imgs, rand_inds)
            pres_acts = tf.gather(acts_flat, rand_inds)
            obj = objectives.Objective.sum(
                [objectives.direction(layer, pres_acts[n], batch=n)
                 for n in range(batch_size)
                 ])

            # Actually do the optimization...
            T = render.make_vis_T(self.model, obj, param_f=pres_imgs)
            tf.global_variables_initializer().run()

            for i in range(n_steps):
                T("vis_op").run()
                if (i+1) % (n_steps//2) == 0:
                    show(pres_imgs.eval()[::4])

            vis_imgs = opt_imgs.eval()

        # Combine the images and display the resulting grid
        print("")
        vis_imgs_ = vis_imgs.reshape(list(acts.shape[:2]) + [W, W, 3])
        vis_imgs_cropped = vis_imgs_[:, :, 2:-2, 2:-2, :]
        show(np.hstack(np.hstack(vis_imgs_cropped)))
        return vis_imgs_cropped
