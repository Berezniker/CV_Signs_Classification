import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


def calculate_gradient(image):
    # gradient calculation with filter core [[-1, 0, 1]]
    grad_x = np.empty(image.shape)
    grad_x[0, :], grad_x[-1, :] = image[1, :], -image[-2, :]
    grad_x[1:-1, :] = image[2:, :] - image[:-2, :]
    grad_y = np.empty(image.shape)
    grad_y[:, 0], grad_y[:, -1] = image[:, 1], -image[:, -2]
    grad_y[:, 1:-1] = image[:, 2:] - image[:, :-2]

    gradient_modulus = np.hypot(grad_x, grad_y)
    gradient_direction = np.rad2deg(np.arctan2(grad_y, grad_x))  # [0..360]

    return gradient_modulus, gradient_direction


def histograms(magnitude, direction, cellRows, cellCols, n_cells_row, n_cells_col, binCount):
    # cell histogram calculation
    histogram = np.zeros((n_cells_row, n_cells_col, binCount))
    for i in range(n_cells_row):
        for j in range(n_cells_col):
            cell_grad = magnitude[i * cellRows: (i + 1) * cellRows, j * cellCols: (j + 1) * cellCols]
            cell_dir = direction[i * cellRows: (i + 1) * cellRows, j * cellCols: (j + 1) * cellCols]
            cell_mask = (cell_dir * binCount / 360).astype(int) % binCount
            for bin_pos in range(binCount):
                histogram[i, j, bin_pos] = np.sum(cell_grad[cell_mask == bin_pos])

    return histogram


def normalize_block(block, eps=1e-5, norm='L2-Hys'):
    if norm == 'L2':
        return block / np.sqrt(np.sum(block ** 2) + eps)
    elif norm == 'L2-Hys':  # on tests showed a better result than 'L2'
        norm_block = block / np.sqrt(np.sum(block ** 2) + eps)
        norm_block = np.clip(norm_block, a_min=0, a_max=0.2)
        return norm_block / np.sqrt(np.sum(norm_block ** 2) + eps)


def build_descriptor(orientation_histogram, n_cells_row, n_cells_col):
    descriptor = np.array([])
    sliding_window_size = [(4, 4)]  # the ability to specify several sizes for thw windows
    for blockRowCells, blockColCells in sliding_window_size:
        n_blocks_row = (n_cells_row - blockRowCells) + 1
        n_blocks_col = (n_cells_col - blockColCells) + 1
        step_row, step_col = 2, 2  # the ability to change block step
        for i in range(0, n_blocks_row, step_row):
            for j in range(0, n_blocks_col, step_col):
                block = orientation_histogram[i: i + blockRowCells, j: j + blockColCells, :]
                descriptor = np.hstack((descriptor, normalize_block(block).ravel()))

    return descriptor


def extract_hog(image):
    """ Функция извлечения признаков HOG """
    img_row, img_col, cellRows, cellCols, binCount = 64, 64, 8, 8, 8
    n_cells_row, n_cells_col = int(img_row // cellRows), int(img_col // cellCols)

    thumbnail = resize(rgb2gray(image), (img_row, img_col))
    grad_mod, grad_dir = calculate_gradient(thumbnail)
    orientation_histogram = histograms(grad_mod, grad_dir, cellRows, cellCols, n_cells_row, n_cells_col, binCount)
    hog_descriptor = build_descriptor(orientation_histogram, n_cells_row, n_cells_col)

    return hog_descriptor
