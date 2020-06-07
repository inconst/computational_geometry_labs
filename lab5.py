
import numpy as np
import matplotlib
from scipy.linalg import null_space
from lab2 import visualize_correspondences

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['figure.figsize'] = 20, 20

# load images
left = mpimg.imread('images/Home3/left.png')
right = mpimg.imread('images/Home3/right.png')
assert left.shape == right.shape, 'Images have different size'
print('Loaded images. Image shape: ', left.shape)
left = (left * 255.0).astype(np.int32)
right = (right * 255.0).astype(np.int32)
H, W, C = left.shape

# manual correspondences (X1 - left image coordinates, X2 - right image coordinates)
X1 = np.array([[883, 607], [980, 466], [367, 430], [631, 334]])
X2 = np.array([[619, 576], [693, 446], [97, 434], [393, 330]])
visualize_correspondences(left, right, X1, X2)

X1 = np.append(X1, np.ones([X1.shape[0], 1], dtype=np.int32), axis=1)
X2 = np.append(X2, np.ones([X2.shape[0], 1], dtype=np.int32), axis=1)
X = np.zeros((8, 9))
for i in range(4):
    X[2 * i, :] = np.concatenate((-X2[i, :], np.zeros(3), X1[i, 0] * X2[i, :]))
    X[2 * i + 1, :] = np.concatenate((np.zeros(3), -X2[i, :], X1[i, 1] * X2[i, :]))
homography = null_space(X).reshape(3, 3)
homography_inv = np.linalg.inv(homography)

new_img = np.concatenate((left, np.zeros_like(left)), axis=1)
print('Concatenating images ..........................')
for y in range(H):
    for x in range(2*W):
        coord = np.dot(homography_inv, [x, y, 1])
        coord = np.round((coord / coord[2]))[:2].astype(np.int32)
        if 0 <= coord[0] < W and 0 <= coord[1] < H:
            if x >= W:
                new_img[y, x, :] = right[coord[1], coord[0], :]
            else:
                new_img[y, x, :] = (new_img[y, x, :] + right[coord[1], coord[0], :]) / 2

plt.figure()
plt.imshow(new_img)
plt.savefig('test.png')
plt.show()

