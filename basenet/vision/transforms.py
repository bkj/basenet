#!/usr/bin/env python

"""
    vision.py
"""

import numpy as np
from PIL import Image
from torchvision import transforms

dataset_stats = {
    "cifar10" : {
        "mean" : (0.4914, 0.4822, 0.4465),
        "std"  : (0.24705882352941178, 0.24352941176470588, 0.2615686274509804),
    },
}


def ReflectionPadding(margin=(4, 4)):
    
    def _reflection_padding(x):
        x = np.asarray(x)
        x = np.pad(x, [(margin[0], margin[0]), (margin[1], margin[1]), (0, 0)], mode='reflect')
        return Image.fromarray(x)
    
    return transforms.Lambda(_reflection_padding)


def NormalizeDataset(dataset='cifar10'):
    assert dataset in set(['cifar10']), 'unknown dataset %s' % dataset
    
    if dataset == 'cifar10':
        return transforms.Normalize(dataset_stats['cifar10']['mean'], dataset_stats['cifar10']['std'])