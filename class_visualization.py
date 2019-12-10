"""Class visualization.

Generation of an image corresponding to a category, in order to visualize the type of
patterns detected by the network.
Based on the articles from:
- Simonyan et al. (2014): https://arxiv.org/abs/1312.6034
- Yosinski et al. (2015): https://arxiv.org/abs/1506.06579
"""

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import *


def create_class_visualization(target_y, model, dtype, init_img=None, l2_reg=1e-3,
                               learning_rate=5, num_iterations=200, blur_every=10,
                               blur_width=0.5, max_jitter=16, clamp=True,
                               show_every=25, class_names=None, savepath=None):
    """
    Generate an image to maximize the score of target_y under a pretrained model.
    Shows the progress every `show_every` iterations.
    Args:
        target_y (int): integer in the range [0, 1000) giving the index of the class
        model (torch.nn.Module): a pretrained CNN that will be used to generate
            the image
        dtype : torch datatype to use for computations

        init_img (torch.Tensor): initial image to use (if None, will be random)
        l2_reg (float): strength of L2 regularization on the image
        learning_rate (float): how big of a step to take
        num_iterations (int): how many iterations to use
        blur_every (int): how often to blur the image as an implicit regularizer
            (if None, does not blur the image)
        blur_width (float): the width of the blurring filter applied
            (if `blur_every` is None, it is ignored)
        max_jitter (int): maximum value to randomly jitter the image as an implicit
            regularizer (if None, does not jitter the image)
        clamp (int): if True, clamps the image to standardize it a bit as an implicit
            regularizer
        show_every (int): How often to show the intermediate result

    Returns:
        (torch.Tensor): the final generated image
    """
    model.type(dtype)

    # randomly initialize the image as a PyTorch Tensor
    if init_img is None:
        img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype)
    else:
        img = init_img.clone().mul_(1.0).type(dtype)
    img.requires_grad = True

    target_y_tensor = torch.tensor([target_y])
    for t in range(num_iterations):
        # randomly jitter the image a bit; this gives slightly nicer results
        if max_jitter:
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.data.copy_(jitter(img.data, ox, oy))

        # Use the model to compute the gradient of the score for the
        # class target_y with respect to the pixels of the image, and make a
        # gradient step on the image using the learning rate.

        out = model(img)
        objective = out[0, target_y] - l2_reg * torch.norm(img)
        objective.backward()
        img = img + learning_rate * img.grad
        img.detach_().requires_grad_()
        # undo the random jitter
        if max_jitter:
            img.data.copy_(jitter(img.data, -ox, -oy))

        # as regularizer, clamp and periodically blur the image
        if clamp:
            for c in range(3):
                lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
                hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
                img.data[:, c].clamp_(min=lo, max=hi)
        if blur_every and t % blur_every == 0:
            img.data = blur_image(img.data.cpu().numpy(), sigma=blur_width)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            # compute the confidence in target_y
            target_y_score = F.softmax(out, 1)[0, target_y]

            plt.imshow(deprocess(img.clone().cpu()))
            class_name = class_names[target_y] if class_names else ''
            plt.title("{}\nIteration {} \nConfidence: {:.2%}"
                      .format(class_name, t + 1, target_y_score))
            plt.gcf().set_size_inches(6, 6)
            plt.gcf().tight_layout()
            plt.axis('off')
            if t != num_iterations - 1:
                plt.show()

    if savepath: plt.savefig(savepath, bbox_inches='tight')
    plt.show()

    return deprocess(img.cpu())
