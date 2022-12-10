import torch


def normalize_features(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** 0.5
