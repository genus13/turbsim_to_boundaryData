
'''
    If the number of TurbSim grid points in y-direction is even an
    additional layer is added in the middle (in such a way the
    extremities of y have always the same values if
    PERIODIC is turned on)
'''

import numpy as np

def interpCenter(matrix):
    prev_rows = matrix[matrix.shape[0] // 2 - 1: matrix.shape[0] // 2]
    next_rows = matrix[matrix.shape[0] // 2: matrix.shape[0] // 2 + 1]
    means = np.mean(np.vstack((prev_rows, next_rows)), axis=0)
    mid_index = matrix.shape[0] // 2
    new_matrix = np.insert(matrix, mid_index, means, axis=0)
    return new_matrix