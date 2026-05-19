import numpy as np


def get_client_stats(y):
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))

    return {
        "num_samples": len(y),
        "class_0": dist.get(0, 0),
        "class_1": dist.get(1, 0)
    }
