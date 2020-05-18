

import numpy as np
import os
from scipy.linalg import null_space
from lab1 import get_disparity_maps
from lab2 import visualize_epipolar_lines, create_point_correspondences, calculate_fundamental_matrix
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


def visualize_rectified(left, right, title="Rectified images", img_offset=20):
    W, H, C = left.shape
    fig, ax = plt.subplots()
    ax.set_title(title)
    left = np.concatenate((left, np.zeros((W, img_offset, 3), dtype=left.dtype)), axis=1)
    plt.imshow(np.concatenate((left, right), axis=1))
    plt.show()


def rectify_images(left, right, F, disparity_map):
    H, W, C = left.shape
    e1, e2 = null_space(F.T).flatten(), null_space(F).flatten()
    if e1[2] < 0:
        e1 = -e1
    if e2[2] < 0:
        e2 = -e2
    e2_norm = e2 / np.linalg.norm(e2)

    # calculate right rectifying matrix
    k = np.array([0, 0, 1], dtype=np.float64)
    T = np.stack((np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([-W / 2, -H / 2, 1])), axis=1)
    R = np.stack((-np.cross(k, np.cross(k, e2_norm)), np.cross(k, e2_norm), k), axis=1).T    # TODO det(R) = -1 ???
    x = np.dot(R, np.dot(T, e1))
    G = np.stack((np.array([1, 0, -x[2] / x[0]]), np.array([0, 1, 0]), np.array([0, 0, 1])), axis=1)
    R_r = np.dot(np.linalg.inv(T), np.dot(G, np.dot(R, T)))
    R_r_inv = np.linalg.inv(R_r)

    # calculate left rectifying matrix
    e2_cross = np.stack((np.array([0, -e2[2], e2[1]]), np.array([e2[2], 0, -e2[0]]), np.array([-e2[1], e2[0], 0])), axis=0)
    M = np.dot(R_r, np.dot(e2_cross, F))
    v = np.cross(M[1, :], M[2, :]).flatten()
    M = np.stack((-v / np.linalg.norm(v), M[1, :], M[2, :]), axis=0)
    X1, X2 = create_point_correspondences(H, W, disparity_map, filter_points=True)
    X1_homogeneous = np.append(X1, np.ones([X1.shape[0], 1], dtype=np.int32), axis=1)
    X2_homogeneous = np.append(X2, np.ones([X2.shape[0], 1], dtype=np.int32), axis=1)
    A = np.dot(M, X1_homogeneous.T).T
    A /= A[:, 2:3]
    transformed_X2 = np.dot(R_r, X2_homogeneous.T).T
    transformed_X2 /= transformed_X2[:, 2:3]
    b = transformed_X2[:, 0:1]
    least_squares = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b).flatten()
    abc_matrix = np.stack((least_squares, np.array([0, 1, 0]), np.array([0, 0, 1])), axis=0)
    print('abc matrix: \n', abc_matrix)
    R_l = np.dot(abc_matrix, M)
    R_l_inv = np.linalg.inv(R_l)

    # fill rectified images using R_r_inv and R_l_inv
    right_rectified = np.zeros((H, W, C), dtype=np.int32)
    left_rectified = np.zeros((H, W, C), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            r_coord = np.dot(R_r_inv, [x, y, 1])
            r_coord = np.round((r_coord / r_coord[2]))[:2].astype(np.int32)
            if 0 <= r_coord[0] < W and 0 <= r_coord[1] < H:
                right_rectified[y, x, :] = right[r_coord[1], r_coord[0], :]

            l_coord = np.dot(R_l_inv, [x, y, 1])
            l_coord = np.round((l_coord / l_coord[2]))[:2].astype(np.int32)
            if 0 <= l_coord[0] < W and 0 <= l_coord[1] < H:
                left_rectified[y, x, :] = left[l_coord[1], l_coord[0], :]

    random_indices = np.random.choice(X1.shape[0], 5)
    visualize_epipolar_lines(left, right, e1 / e1[2], e2 / e2[2], X1[random_indices], X2[random_indices],
                             title='Before rectification')
    # visualize_rectified(left_rectified, right_rectified)

    disp_h, disp_v = get_disparity_maps(left_rectified, right_rectified, alpha=2, h_norm=1, beta=50, max_disp_x=30,
                                        max_disp_y=0, max_negative_disp_x=30, max_negative_disp_y=0)
    new_disparity_map = np.stack((disp_h, disp_v), axis=-1)
    calculate_fundamental_matrix(left_rectified, right_rectified, new_disparity_map)


if __name__ == '__main__':
    dir_name = 'Home2'
    image_dir = os.path.join('images', dir_name)
    left = np.load(os.path.join(image_dir, 'left.npy'))
    right = np.load(os.path.join(image_dir, 'right.npy'))
    F = np.load(os.path.join(image_dir, 'f_matrix.npy'))
    disparity_map = np.load(os.path.join(image_dir, 'disp_map.npy'))
    rectify_images(left, right, F, disparity_map)
