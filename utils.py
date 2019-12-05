"""Utilities functions"""
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(x_numpy, sigma=1):
    """Blur an image by applying a gaussian filter on it.
    Args:
        x (numpy.array): input image
        sigma (number): standard deviation
    Returns
        torch.Tensor: output blurred tensor image
    """
    x_numpy = gaussian_filter1d(x_numpy, sigma, axis=2)
    x_numpy = gaussian_filter1d(x_numpy, sigma, axis=3)
    return torch.tensor(x_numpy)

def jitter(x, ox, oy):
    """
    Helper function to randomly jitter an image.
    Args:
        x (torch.Tensor): batch of images
        ox (int): number of pixels to jitter along width axis
        oy (int): number of pixels to jitter along height axis
    Returns:
        torch.Tensor: new tensor
    """
    if ox != 0:
        left = x[:, :, :, :-ox]
        right = x[:, :, :, -ox:]
        x = torch.cat([right, left], dim=3)
    if oy != 0:
        top = x[:, :, :-oy]
        bottom = x[:, :, -oy:]
        x = torch.cat([bottom, top], dim=2)
    return x
