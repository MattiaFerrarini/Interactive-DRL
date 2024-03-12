import numpy as np

def contains(list, np_arr):
    return any(np.array_equal(np_arr, arr) for arr in list)

def remove(list, np_arr):
    index_to_remove = next((i for i, arr in enumerate(list) if np.array_equal(arr, np_arr)), None)
    del list[index_to_remove]