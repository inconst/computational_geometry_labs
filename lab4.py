import numpy as np
import os
from exif import Image


def cross_product_matrix(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def calculate_essential_matrix(image_dir):
    left = np.load(os.path.join(image_dir, 'left.npy'))
    right = np.load(os.path.join(image_dir, 'right.npy'))
    F = np.load(os.path.join(image_dir, 'f_matrix.npy'))
    with open(os.path.join(image_dir, 'left_full.jpg'), 'rb') as image_file:
        left_metadata = Image(image_file)
    with open(os.path.join(image_dir, 'right_full.jpg'), 'rb') as image_file:
        right_metadata = Image(image_file)

    H, W, _ = left.shape
    s = 0.8 * 1e-6  # pixel size from camera specification
    s *= left_metadata['pixel_x_dimension'] / W  # account for image resize
    f1 = left_metadata['focal_length'] * 1e-3
    f2 = right_metadata['focal_length'] * 1e-3
    K1 = np.array([[f1 / s, 0, W / 2], [0, f1 / s, H / 2], [0, 0, 1]])
    K2 = np.array([[f2 / s, 0, W / 2], [0, f2 / s, H / 2], [0, 0, 1]])
    E = K2.T @ F @ K1

    U, S, VT = np.linalg.svd(E)
    sigma = (S[0] + S[1]) / 2
    E = U @ np.diag(np.array([sigma, sigma, 0])) @ VT

    # check condition on essential matrix
    cond = 2 * E @ E.T @ E - np.trace(E @ E.T) * E
    assert np.all(np.abs(cond) < 1e-6), 'Condition `2E * E.T * E - np.trace(E * E.T) * E = 0` is not True'

    # find R and c from SVD decomposition
    W = U @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R = np.linalg.det(W) * np.linalg.det(VT) * W @ VT
    c = sigma * np.linalg.det(W) * VT.T[:, 2]
    assert np.all(np.abs(E - R @ cross_product_matrix(c)) < 1e-6), 'Condition E = R * [c]_x is not True'
    print('R matrix: \n', R)
    print('c vector: ', c)


if __name__ == '__main__':
    dir_name = 'Home2'
    image_dir = os.path.join('images', dir_name)
    calculate_essential_matrix(image_dir)
