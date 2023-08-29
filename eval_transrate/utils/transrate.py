import numpy as np


def cal_coding_rate(features, eps=1E-4):
    n, d = features.shape
    _, rate = np.linalg.slogdet(np.eye(d) + 1 / (n * eps) * features.transpose() @ features)
    return 0.5 * rate


def cal_transrate(features, labels, eps=1E-4):
    features = features - np.mean(features, axis=0, keepdims=True)
    RZ = cal_coding_rate(features, eps)
    RZY = 0.
    K = int(labels.max() + 1)
    for i in range(K):
        RZY += cal_coding_rate(features[(labels == i).flatten()], eps)
    return RZ - RZY/K