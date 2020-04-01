from lab1 import get_images_and_disparity_map
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def ransac(X1, X2, max_iterations=100, eps=1.0):
    print('Finding fundamental matrix ........................')
    num_points = X1.shape[0]
    best_k, best_F = 0, np.zeros((3, 3))
    for i in range(max_iterations):
        random_sample = np.random.choice(num_points, 7, replace=False)
        X1_7pts, X2_7pts = X1[random_sample], X2[random_sample]
        X = np.concatenate((X1_7pts[:, 0:1] * X2_7pts, X1_7pts[:, 1:2] * X2_7pts, X2_7pts), axis=1)
        ns = null_space(X).T
        F1, F2 = ns[0, :].reshape(3, 3), ns[1, :].reshape(3, 3)
        det_F1, det_F2 = np.linalg.det(F1), np.linalg.det(F2)
        roots = np.roots([det_F2, det_F2 * np.trace(np.dot(np.linalg.inv(F2), F1)),
                          det_F1 * np.trace(np.dot(np.linalg.inv(F1), F2)), det_F1])
        real_roots = roots[~np.iscomplex(roots)]
        for root in real_roots:
            F = (F1 + root * F2)
            k = np.sum(np.abs(np.sum(np.dot(X1, F) * X2, axis=1)) < eps)
            if k > best_k:
                best_k, best_F = k, F
    print('Best K: ', best_k)
    print('Best F: ', best_F)
    return best_F


def main():
    left, right, disp_horizontal, disp_vertical = get_images_and_disparity_map()
    disparity_map = np.stack((disp_horizontal, disp_vertical), axis=-1)
    W, H, C = left.shape
    X1_points = np.transpose(np.mgrid[0:H:1, 0:W:1], (2, 1, 0))
    X2_points = X1_points.copy()
    X2_points -= disparity_map
    # check if points within image boundaries
    assert not ((X2_points[:, :, 0] < 0) | (X2_points[:, :, 0] >= H) |
                (X2_points[:, :, 1] < 0) | (X2_points[:, :, 1] >= W)).any()

    # filter points
    print('Filtering points ........................')
    filter_mask = np.zeros((W, H), dtype=np.bool)
    filter_neighbors = 1
    for i in range(filter_neighbors, W - filter_neighbors):
        for j in range(filter_neighbors, H - filter_neighbors):
            if not np.array_equal(disparity_map[i, j], np.zeros(2)):
                window = disparity_map[i-filter_neighbors:i+filter_neighbors+1, j-filter_neighbors:j+filter_neighbors+1]
                equal_neighbors = np.sum(np.equal(window, disparity_map[i, j]), axis=2) == 2
                if equal_neighbors.all():
                    filter_mask[i, j] = True
    num_filtered = filter_mask.sum()
    print('Total number of filtered points', num_filtered)
    filtered_X1 = X1_points[filter_mask]
    filtered_X2 = X2_points[filter_mask]

    # find fundamental matrix and epipolar points
    X1_homogeneous = np.append(filtered_X1, np.ones([filtered_X1.shape[0], 1], dtype=np.int32), axis=1)
    X2_homogeneous = np.append(filtered_X2, np.ones([filtered_X2.shape[0], 1], dtype=np.int32), axis=1)
    F = ransac(X1_homogeneous, X2_homogeneous)
    e1, e2 = null_space(F.T), null_space(F)
    e1 /= e1[2]
    e2 /= e2[2]
    print('Epipole left:', e1)
    print('Epipole right:', e2)

    # visualize some point correspondences and epipolar points
    random_indices = np.random.choice(num_filtered, 20)
    random_X1 = filtered_X1[random_indices]
    random_X2 = filtered_X2[random_indices]
    fig, ax = plt.subplots()
    plt.imshow(np.concatenate((left, right), axis=1))
    plt.scatter(e1[0], e1[1], c='red')
    plt.scatter(H + e1[0], e1[1], c='red')
    line_segments = LineCollection([[random_X1[i], np.array([H, 0]) + random_X2[i]] for i in range(random_X1.shape[0])], colors='yellow')
    ax.add_collection(line_segments)
    plt.show()


if __name__ == '__main__':
    main()
