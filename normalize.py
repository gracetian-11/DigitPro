import numpy as np

def scale_to_range(data, left_bound, right_bound):
    return ((data - np.min(data)) / (np.max(data) - np.min(data))) * (right_bound - left_bound) + left_bound, np.min(data), np.max(data)

def unscale_from_range(data, min_val, max_val, left_bound, right_bound): 
    return (data - left_bound) / (right_bound - left_bound) * (max_val - min_val) + min_val

"""
def unscale_from_range(data, original_data, left_bound, right_bound):
    min_val = float('inf')
    max_val = float('-inf')
    for i in range(len(original_data)):
        for j in range(len(original_data[i])):
            min_val = min(min_val, np.min(original_data[i][j]))
            max_val = max(max_val, np.max(original_data[i][j]))
    return ((data - left_bound) / (right_bound - left_bound)) * (max_val - min_val) + min_val
"""

"""
TESTING
"""
# data = [[5, 8, -1, 3], [-10, 2, -7, 6], [4, 1, -9, 5]]
# data = [[5], [8], [-1], [3]]
# data = np.array(data)
# scaled_data, d_min, d_max = scale_to_range(data, -1, 1)
# unscaled_data = unscale_from_range(scaled_data, d_min, d_max, -1, 1)
# print(data)
# print(scaled_data)
# print(unscaled_data)

