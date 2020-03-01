import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

image_files = {
    'cones': {'left': 'images/Cones/view1.png', 'right': 'images/Cones/view2.png'},
    'drumsticks': {'left': 'images/Drumsticks/view1.png', 'right': 'images/Drumsticks/view2.png'},
    'test': {'left': 'images/Test/left.png', 'right': 'images/Test/right.png'}
}
default_image = 'test'

parser = argparse.ArgumentParser()
parser.add_argument('--left_image', type=str, default=image_files[default_image]['left'], help='Path to left image')
parser.add_argument('--right_image', type=str, default=image_files[default_image]['right'], help='Path to right image')
parser.add_argument('--max_displacement_x', type=int, default=20, help='Maximum displacement value over x axis')
parser.add_argument('--max_displacement_y', type=int, default=5, help='Maximum displacement value over y axis')
parser.add_argument('--max_negative_displacement_x', type=int, default=0,
                    help='Maximum negative displacement value over x axis')
parser.add_argument('--max_negative_displacement_y', type=int, default=5,
                    help='Maximum negative displacement value over y axis')
parser.add_argument('--h_norm', type=int, default=1, help='Order of norm in h function')
parser.add_argument('--alpha', type=int, default=2, help='Smoothing alpha parameter')
parser.add_argument('--beta', type=int, default=30, help='Smoothing beta parameter')
args = parser.parse_args()


def h_func(L, R, norm=args.h_norm, axis=2):
    return np.linalg.norm(L - R, norm, axis).astype(np.int32)


def g_func(d1, d2, beta=args.beta):
    return np.minimum(beta, np.abs(d1 - d2)) if beta is not None else np.abs(d1 - d2)


def expanded_array(a, repeat_num, axis=-1):
    return np.repeat(np.expand_dims(a, axis=axis), repeat_num, axis=axis)


def get_displacement(left_image, right_image,
                     alpha=args.alpha,
                     max_disp_x=args.max_displacement_x,
                     max_disp_y=args.max_displacement_y,
                     max_negative_disp_x=args.max_negative_displacement_x,
                     max_negative_disp_y=args.max_negative_displacement_y):
    print('Generating displacement map ..........................')
    start_time = time.time()
    current_time = time.time()
    W, H, C = left_image.shape  # W - width, H - height, C - number of channels
    D = (max_disp_x + max_negative_disp_x + 1) * (max_disp_y + max_negative_disp_y + 1)
    displacement_map = np.zeros((W, H), dtype=np.int32)

    # NOTE: requires lots of memory
    L_expanded = expanded_array(left_image, D)
    R_shifted = np.full((W, H, C, D), 10000000, dtype=np.int32)  # initialize with infinity
    k = 0
    for d_x in range(-max_negative_disp_x, max_disp_x + 1):
        for d_y in range(-max_negative_disp_y, max_disp_y + 1):
            if d_x >= 0:
                if d_y >= 0:
                    R_shifted[d_y:, d_x:, :, k] = right_image[:-d_y if d_y != 0 else W, :-d_x if d_x != 0 else H, :]
                else:
                    R_shifted[:d_y, d_x:, :, k] = right_image[-d_y:, :-d_x if d_x != 0 else H, :]
            else:
                if d_y >= 0:
                    R_shifted[d_y:, :d_x, :, k] = right_image[:-d_y if d_y != 0 else W, -d_x:, :]
                else:
                    R_shifted[:d_y, :d_x, :, k] = right_image[-d_y:, -d_x:, :]
            k += 1
    h_table = h_func(L_expanded, R_shifted, axis=2)  # h_table.shape = [W, H, D]
    print('Created h_table. Time elapsed: {:.2f} seconds'.format(time.time() - current_time))
    current_time = time.time()

    g_table = np.fromfunction(g_func, (D, D), dtype=np.int32)
    g_table = expanded_array(g_table, H, axis=0)  # g_table.shape = [H, D, D]
    print('Created g_table. Time elapsed: {:.2f} seconds'.format(time.time() - current_time))
    current_time = time.time()

    # fill dynamic programming table
    DP_table = np.zeros((W, H, D), dtype=np.int32)
    path_table = np.zeros((W - 1, H, D), dtype=np.int32)
    DP_table[0, :, :] = h_table[0, :, :]
    for x in range(1, W):
        last_sum = np.expand_dims(DP_table[x - 1, :, :], axis=-1) + alpha * g_table  # [H, D] + [H, D, D] -> [H, D, D]
        min_indices = np.argmin(last_sum, axis=1)
        min_values = np.min(last_sum, axis=1)

        path_table[x - 1, :, :] = min_indices
        DP_table[x, :, :] = h_table[x, :, :] + min_values
    print('Filled DP_table. Time elapsed: {:.2f} seconds'.format(time.time() - current_time))

    # restore optimal path
    min_path_indices = np.argmin(DP_table[W - 1, :, :], axis=1)
    displacement_map[W - 1, :] = min_path_indices
    for x in reversed(range(W - 1)):
        min_path_indices = path_table[x, :, :][np.arange(H), min_path_indices]
        displacement_map[x, :] = min_path_indices

    print('Finished generating displacement map. Total time elapsed: {:.2f} seconds'.format(time.time() - start_time))
    return displacement_map


left = mpimg.imread(args.left_image)
right = mpimg.imread(args.right_image)
assert left.shape == right.shape, 'Images have different size'
print('Loaded images. Image shape: ', left.shape)
left = (left * 255.0).astype(np.int32)
right = (right * 255.0).astype(np.int32)

f = plt.figure(figsize=(20, 20))
f.add_subplot(1, 3, 1)
plt.imshow(left)
f.add_subplot(1, 3, 2)
plt.imshow(right)
disp = get_displacement(left, right)
disp = disp.astype(np.float32) / np.max(disp) if np.max(disp) != 0 else disp
disp = expanded_array(disp, 3)
f.add_subplot(1, 3, 3)
plt.imshow(disp)
plt.show()
