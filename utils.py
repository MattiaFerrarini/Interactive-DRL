import numpy as np

# check if list contains np_arr of type ndarray
def contains(list, np_arr):
    return any(np.array_equal(np_arr, arr) for arr in list)

# removes np_arr (ndarray) from list
def remove(list, np_arr):
    index_to_remove = next((i for i, arr in enumerate(list) if np.array_equal(arr, np_arr)), None)
    if index_to_remove != None:
        del list[index_to_remove]