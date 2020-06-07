import matplotlib
import numpy as np
from scipy.linalg import null_space

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os


def ransac(X1, X2, max_iterations=10000, eps=0.001):
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
        if np.abs(det_F1) < 1e-8 or np.abs(det_F2) < 1e-8:
            F = F1 if np.abs(det_F1) < 1e-8 else F2
            k = np.sum(np.abs(np.einsum('ij,ji->i', np.dot(X1, F.T), X2.T)) < eps)
            if k > best_k:
                best_k, best_F = k, F
        else:
            roots = np.roots([det_F2, det_F2 * np.trace(np.dot(np.linalg.inv(F2), F1)),
                              det_F1 * np.trace(np.dot(np.linalg.inv(F1), F2)), det_F1])
            real_roots = roots[~np.iscomplex(roots)]
            for root in real_roots:
                F = (F1 + root * F2)
                k = np.sum(np.abs(np.einsum('ij,ji->i', np.dot(X1, F.T), X2.T)) < eps)
                if k > best_k:
                    best_k, best_F = k, F
    print('Best K: ', best_k)
    print('Best F: ', best_F)
    return best_F


def create_point_correspondences(H, W, disparity_map=None, filter_points=True):
    X1_points = np.transpose(np.mgrid[0:W:1, 0:H:1], (2, 1, 0))
    X2_points = X1_points.copy()
    if disparity_map is not None:
        X2_points -= disparity_map
    # check if points within image boundaries
    assert not ((X2_points[:, :, 0] < 0) | (X2_points[:, :, 0] >= W) |
                (X2_points[:, :, 1] < 0) | (X2_points[:, :, 1] >= H)).any()

    if filter_points:
        print('Filtering points ........................')
        filter_mask = np.zeros((H, W), dtype=np.bool)
        filter_neighbors = 1
        for i in range(filter_neighbors, H - filter_neighbors):
            for j in range(filter_neighbors, W - filter_neighbors):
                if not np.array_equal(disparity_map[i, j], np.zeros(2)):
                    window = disparity_map[i - filter_neighbors:i + filter_neighbors + 1,
                             j - filter_neighbors:j + filter_neighbors + 1]
                    equal_neighbors = np.sum(np.equal(window, disparity_map[i, j]), axis=2) == 2
                    if equal_neighbors.all():
                        filter_mask[i, j] = True
        X1 = X1_points[filter_mask]
        X2 = X2_points[filter_mask]
    else:
        X1 = X1_points.reshape((-1, 2))
        X2 = X2_points.reshape((-1, 2))

    return X1, X2


def calculate_fundamental_matrix(left, right, disparity_map):
    H, W, C = left.shape
    X1, X2 = create_point_correspondences(H, W, disparity_map, filter_points=True)
    print('Total number of filtered points', len(X1))

    # find fundamental matrix and epipolar points
    X1_homogeneous = np.append(X1, np.ones([X1.shape[0], 1], dtype=np.int32), axis=1)
    X2_homogeneous = np.append(X2, np.ones([X2.shape[0], 1], dtype=np.int32), axis=1)
    F = ransac(X1_homogeneous, X2_homogeneous)

    e1, e2 = null_space(F.T), null_space(F)
    e1 /= e1[2]
    e2 /= e2[2]
    print('Epipole left:', e1)
    print('Epipole right:', e2)

    # visualize some point correspondences and epipolar points
    random_indices = np.random.choice(len(X1), 5)
    random_X1, random_X2 = X1[random_indices], X2[random_indices]
    visualize_correspondences(left, right, random_X1, random_X2)
    visualize_epipolar_lines(left, right, e1, e2, random_X1, random_X2)

    return F


def visualize_correspondences(left, right, X1, X2, title="Point correspondences"):
    _, W, _ = left.shape
    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.imshow(np.concatenate((left, right), axis=1))
    line_segments = LineCollection([[X1[i], np.array([W, 0]) + X2[i]] for i in range(X1.shape[0])],
                                   colors='yellow')
    ax.add_collection(line_segments)
    plt.show()


def visualize_epipolar_lines(left, right, e1, e2, x1, x2, title="Epipolar lines", img_offset=20):
    H, W, C = left.shape
    fig, ax = plt.subplots()
    ax.set_title(title)
    left = np.concatenate((left, np.zeros((H, img_offset, 3), dtype=left.dtype)), axis=1)
    plt.imshow(np.concatenate((left, right), axis=1))

    def image_border_intersection(p1, p2, W, H, offset=np.array([0, 0])):
        # calculates intersection of line between points p1 and p2 with image borders
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = (p1[0] - p2[0]) * p1[1] + (p2[1] - p1[1]) * p1[0]
        n = np.array([a, b, c]).squeeze()

        left_vertical, right_vertical = np.array([1, 0, 0]), np.array([1, 0, -W])
        top_horizontal, bottom_horizontal = np.array([0, 1, -H]), np.array([0, 1, 0])

        valid_intersections = []
        for line in [left_vertical, right_vertical, top_horizontal, bottom_horizontal]:
            intersection = np.cross(n, line)
            intersection = (intersection / intersection[2]).astype(np.int32)
            if 0 <= intersection[0] <= W and 0 <= intersection[1] <= H:
                valid_intersections.append(intersection[:2] + offset)
        return valid_intersections

    lines_1 = [image_border_intersection(e1, x, W - 1, H - 1) for x in x1]
    lines_2 = [image_border_intersection(e2, x, W - 1, H - 1, np.array([W + img_offset, 0])) for x in x2]
    lines = lines_1 + lines_2
    line_segments = LineCollection(lines, colors='yellow')
    ax.add_collection(line_segments)

    plt.show()


if __name__ == '__main__':
    dir_name = 'Home2'
    image_dir = os.path.join('images', dir_name)
    disparity_map = np.load(os.path.join(image_dir, 'disp_map.npy'))
    left = np.load(os.path.join(image_dir, 'left.npy'))
    right = np.load(os.path.join(image_dir, 'right.npy'))
    F_matrix = calculate_fundamental_matrix(left, right, disparity_map)
    np.save(os.path.join(image_dir, 'f_matrix.npy'), F_matrix)
