"""Adversarial examples or fooling examples.

Based on the articles from:
- Szegedy et al. (2014): https://arxiv.org/abs/1312.6199
- Goodfellow et al. (2015): https://arxiv.org/abs/1412.6572
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import *

def make_fooling_image(x, destination_y, model, max_iters=100, confidence=None,
                       optimization='class_output', plot_progress=False):
    """
    Generate a fooling image that is close to x, but that the model classifies
    as destination_y.
    Args:
        x (torch.Tensor): input image
        dest_y (int or torch.Tensor): the target class, an integer in the range
            [0, 1000)
        model (torch.nn.Module): a pretrained CNN
        confidence (float, optional): stop when the fooling confidence is over this number
            in the interval [0, 1[
        optimization (str): The type of optimization performed on the image, i.e. what's the
            objective.
            - if 'class_output', the output of the network for target class in maximized
                with gradient ascent;
            - if 'cross_entropy_loss', the cross-entropy loss between the output of the
                network and the target class is minimized with gradient descent.

        plot_progress (bool): if True, plot progress at every iteration

    Returns:
        torch.Tensor: an image that is close to the input, but that is classifed
            as destination_y by the model.
        int: the number of gradient iterations it took
    """
    # Generate a fooling image x_fooling that the model will classify as
    # the class target_y. You should perform gradient ascent on the score of the
    # target class, stopping when the model is fooled.
    # When computing an update step, first normalize the gradient:
    #   g_norm = g / ||g||_2

    # Note: For most examples, you should be able to generate a fooling image
    # in fewer than 100 iterations of gradient ascent.
    # You can print your progress over iterations to check your algorithm.

    if type(destination_y) is int:
        dest_y = torch.tensor([destination_y])
    else:
        dest_y = destination_y

    # initialize our fooling image to the input image
    x_fooling = x.clone().requires_grad_()
    # a learning rate of 0.5 works well in practice
    learning_rate = 0.5
    t = 1
    out = model(x_fooling)
    for t in range(0, max_iters):

        if optimization == 'class_output':
            # compute objective
            objective = out[0, destination_y]
            objective.backward()
            x_fooling = x_fooling + learning_rate * \
                (x_fooling.grad / torch.norm(x_fooling.grad))

        elif optimization == 'cross_entropy_loss':
            loss = F.cross_entropy(out, dest_y)
            loss.backward()
            x_fooling = x_fooling - learning_rate * \
                (x_fooling.grad / torch.norm(x_fooling.grad))
        else:
            raise ValueError('{}: Unknown type of optimization'.format(optimization))

        x_fooling.detach_().requires_grad_()
        # compute raw outputs
        out = model(x_fooling)

        # compute the predicted label and the confidence in destination_y
        pred_y = out.argmax(1)
        dest_y_score = F.softmax(out, 1)[0, destination_y].item()

        if plot_progress:
            # plot fooling image and its modifications
            plt.subplot(1, 2, 1)
            plt.imshow(np.asarray(deprocess(x_fooling.clone())).astype(np.uint8))
            plt.title("Fooled image \nIteration {} \nConfidence: {:.2%}"
                      .format(t+1, dest_y_score))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(np.asarray(deprocess(10* (x_fooling - x), should_rescale=False)))
            plt.title("Magnified difference (10x)")
            plt.axis('off')
            plt.show()

        if pred_y.eq(dest_y):
            if confidence is None:
                break
            elif dest_y_score > confidence:
                break

    if t == max_iters - 1:
        print('Not enough iterations to obtain a fooling example.')

    return x_fooling, t+1
