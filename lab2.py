from lab1 import get_images_and_disparity_map
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

'''
+ 1. Refactor Lab1 to work with Lab2                 
+ 2. Get disparity map  
+ 3. Recover x2 from disparity map                             
4. Filter point pairs
+ 5. Visualize disparities
+ 6. Use RANSAC for points from disparity map
7. Visualize epipolar lines
8. Experiment with parameters, find edge cases

'''


def ransac(X1, X2, max_iterations=100, eps=1.0):
    X1 = np.append(X1, np.ones([X1.shape[0], 1], dtype=np.int32), axis=1)
    X2 = np.append(X2, np.ones([X2.shape[0], 1], dtype=np.int32), axis=1)
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
                print('new k: ', k)
                best_k, best_F = k, F
    return best_F


def main():
    left, right, disp_horizontal, disp_vertical = get_images_and_disparity_map()
    disparity_map = np.stack((disp_horizontal, disp_vertical), axis=-1)
    W, H, C = left.shape
    X1_points = np.transpose(np.mgrid[0:H:1, 0:W:1], (2, 1, 0))
    X2_points = X1_points.copy()
    X2_points -= disparity_map
    assert not ((X2_points[:, :, 0] < 0) | (X2_points[:, :, 0] >= H) |
                (X2_points[:, :, 1] < 0) | (X2_points[:, :, 1] >= W)).any()  # check if points within boundaries

    random_indices = np.random.choice(W*H, 100)
    random_X1 = X1_points.reshape(-1, 2)[random_indices]
    random_X2 = X2_points.reshape(-1, 2)[random_indices]
    F = ransac(random_X1, random_X2)

    fig, ax = plt.subplots()
    plt.imshow(left)
    line_segments = LineCollection([[random_X1[i], random_X2[i]] for i in range(random_X1.shape[0])], colors='yellow')
    ax.add_collection(line_segments)
    plt.show()
    '''
    Filtering:
    + 1. Transform X1 points to X2 points                                         
    + 2. Assert boundaries
    3. Eliminate duplicates (preserving order and indexing, probably with maps)
    4. Find clusters of same points (that don't overlap), eliminate them
    5. Pick only non-masked points
    '''


if __name__ == '__main__':
    main()
    # X1 = np.random.randint(0, 300, (10000, 2))
    # X2 = np.random.randint(0, 300, (10000, 2))
    # ransac(X1, X2)
