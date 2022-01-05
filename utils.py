import numpy as np 
from sklearn.datasets import make_blobs
from numpy.random import uniform, normal

def _generate_2d_rotation(theta=0):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return R

def generate_linear_gaussians(
    n_samples,
    theta=0,
    centers=None,
    class_label=None,
    cluster_std=1,
    random_state=None,
):
    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-1, 0), (1, 0)])

    if class_label == None:
        class_label = [0, 1]

    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )

    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=1,
        centers=centers,
        cluster_std=cluster_std,
        # center_box=center_box,
    )

    for blob in range(blob_num):
        y[np.where(y == blob)] = class_label[blob]

    if theta != 0:
        R = _generate_2d_rotation(theta)
        X = X @ R
    
    return X, y.astype(int)
