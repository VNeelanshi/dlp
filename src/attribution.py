import lucid.modelzoo.vision_models as models
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.showing import _image_url, _display_html
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.utils import googlenet_spritemap

""" This module contains spatial attribution and channel attribution code"""


# show distribution of attribution magnitudes
def visualize_attr_distribution(attr):
    # the [::-1] just reverses the sorted array so it goes from highest (pos) to lowest (neg)
    # this is a simple bar chart
    y = np.sort(attr)[::-1]
    x = np.arange(len(y))
    plt.bar(x, y)
    plt.ylabel("Attribution Values")
    plt.title("Distribution of Attribution")
    plt.show()

# a technique for exploring how different detectors in the network affect its output
class ChannelAttribution():
    def __init__(self, model):
        self.model = model

    def score_f(self, logit, name):
        if name is None:
            return 0
        elif name == "logsumexp":
            base = tf.reduce_max(logit)
            return base + tf.log(tf.reduce_sum(tf.exp(logit - base)))
        elif name in self.model.labels:
            return logit[self.model.labels.index(name)]
        else:
            raise RuntimeError("Unsupported")

    def _present_results(self, layer, channel_attr, n_show):
        # Now we just need to present the results.
        # Get spritemaps
        spritemap_n, spritemap_url = googlenet_spritemap(layer)

        # Let's show the distribution of attributions
        print("Distribution of attribution across channels:")
        print("")

        visualize_attr_distribution(channel_attr)

        # Let's pick the most extreme channels to show
        ns_pos = list(np.argsort(-channel_attr)[:n_show])
        ns_neg = list(np.argsort(channel_attr)[:n_show][::-1])

        # TODO: replace this visualization
        # ...  and show them with ChannelAttrWidget
        print("")
        print("Top", n_show, "channels in each direction:")
        print("")
        # lucid_svelte.ChannelAttrWidget({
        #   "spritemap_url": spritemap_url,
        #   "sprite_size": 110,
        #   "sprite_n_wrap": spritemap_n,
        #   "attrsPos": [{"n": n, "v": str(float(channel_attr[n]))[:5]} for n in ns_pos],
        #   "attrsNeg": [{"n": n, "v": str(float(channel_attr[n]))[:5]} for n in ns_neg]
        # })

    def channel_attr_simple(self, img, layer, class1, class2, n_show=4):
        # Set up a graph for doing attribution...
        with tf.Graph().as_default(), tf.Session() as sess:
            t_input = tf.placeholder_with_default(img, [None, None, 3])
            T = render.import_model(self.model, t_input, t_input)

            # Compute activations
            acts = T(layer).eval()

            # Compute gradient
            logit = T("softmax2_pre_activation")[0]
            score = self.score_f(logit, class1) - self.score_f(logit, class2)
            t_grad = tf.gradients([score], [T(layer)])[0]
            grad = t_grad.eval()

            # Let's do a very simple linear approximation attribution.
            # That is, we say the attribution of y to x is
            # the rate at which x changes y times the value of x.
            attr = (grad * acts)[0]
            # Then we reduce down to channels.
            channel_attr = attr.sum(0).sum(0)
        self._present_results(layer, channel_attr, n_show)

    # integrate on a path along activations
    def channel_attr_path(self, img, layer, class1, class2, n_show=4, stochastic_path=False, N=100):
        # Set up a graph for doing attribution...
        with tf.Graph().as_default(), tf.Session() as sess:
            t_input = tf.placeholder_with_default(img, [None, None, 3])
            T = render.import_model(self.model, t_input, t_input)

            # Compute activations
            acts = T(layer).eval()

            # Compute gradient
            logit = T("softmax2_pre_activation")[0]
            score = self.score_f(logit, class1) - self.score_f(logit, class2)
            t_grad = tf.gradients([score], [T(layer)])[0]

            # Integrate on a path from acts=0 to acts=acts
            attr = np.zeros(acts.shape[1:])
            for n in range(N):
                acts_ = acts * float(n) / N
                if stochastic_path:
                    acts_ *= (np.random.uniform(0, 1, [528]) + np.random.uniform(0, 1, [528])) / 1.5
                grad = t_grad.eval({T(layer): acts_})
                attr += 1.0 / N * (grad * acts)[0]
            # Then we reduce down to channels.
            channel_attr = attr.sum(0).sum(0)
        self._present_results(layer, channel_attr, n_show)
