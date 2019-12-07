"""Saliency map.
A saliency map indicates the importance of each pixel in an image to the score of the
true class.
Based on the article from Simonyan et al. (2014) :https://arxiv.org/abs/1312.6034
"""
import torch
import torch.nn.functional as F


def compute_saliency_maps(data, target, model):
    """
    Compute a class saliency map using the model for images data and labels target
    Args:
        data (torch.tensor): input images
        target: (torch.LongTensor): labels
        model (torch.nn.Module): a pretrained CNN that will be used to compute the
            saliency map.
    Returns:
        torch.tensor: a tensor giving the saliency maps for the input images.
    """
    # Perform a forward and backward pass through the model to compute the gradient
    # of the correct class score with respect  to each input image. You first want
    # to compute the loss over the correct scores, and then compute the gradients
    # with a backward pass.

    batch_size = target.size(0)
    data = data.clone().requires_grad_()

    # compute raw outputs
    out = model(data)
    objective = torch.zeros(batch_size)
    for idx in range(batch_size):
        objective[idx] = out[idx, target[idx]]

    objective.backward(torch.ones(batch_size))

    # get saliency maps from gradients of inputs
    saliency = torch.abs(data.grad)
    saliency, _ = saliency.max(1)

    return saliency

    #Note: below is what I was doing before: instead of computing the gradients
    # on the output of the network for each class, I computed them on the cross
    # entropy loss between the output and the classes.
    # This is a different thing, and altough it gives similar results it is not the
    # exact definition of the saliency map.

    # cross entropy loss btw output and target
    # out = model(data)
    # loss = F.cross_entropy(out, target)
    # compute gradients
    # loss.backward()
