def ex4(image_array, offset, spacing):
    import numpy as np
    
    if isinstance(image_array, np.ndarray) == False:
        raise TypeError("image_array is not a numpy array.")

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise NotImplementedError("image_array is not a 3D array or the 3rd dimension is not equal to 3")

    try:
        int(offset[0])
        int(offset[1])
        int(spacing[0])
        int(spacing[1])

    except ValueError:
        raise ValueError("offset or spacing value(s) are not convertible to int objects")

    else:
        pass

    if offset[0] < 0 or offset[1] < 0 or offset[0] > 32 or offset[1] > 32:
        raise ValueError("offset values are less than 0 or greater than 32")

    if spacing[0] < 2 or spacing[1] < 2 or spacing[0] > 8 or spacing[1] > 8:
        raise ValueError("spacing values are smaller than 2 or larger than 8")

    x_start, y_start = int(offset[0]), int(offset[1])
    x_spacing, y_spacing = int(spacing[0]), int(spacing[1])

    M, N, _ = image_array.shape

    good_pixels = [(y, x) for y in range(y_start, M, y_spacing) for x in range(x_start, N, x_spacing)]

    if len(good_pixels) < 144:
        raise ValueError("number of the remaining known image pixels would be smaller than 144")

    input_array = np.zeros_like(image_array)
    known_array = np.zeros_like(image_array)
    known_array_replacement = np.array([1., 1., 1.])
    for y, x in good_pixels:
        input_array[y, x, :] = image_array[y, x, :]
        known_array[y, x, :] = known_array_replacement
    input_array = np.transpose(input_array, (2, 0, 1))
    known_array = np.transpose(known_array, (2, 0, 1))

    target_array = image_array.copy()
    target_array_indices = np.array([True] * M * N * 3).reshape(M, N, 3)
    target_array_indices_replacement = np.array([False, False, False])
    for y, x in good_pixels:
        target_array_indices[y, x, :] = target_array_indices_replacement
    target_array = target_array[target_array_indices]
    target_array = target_array.reshape(-1, 3)
    target_array_R = target_array[:, 0].flatten()
    target_array_G = target_array[:, 1].flatten()
    target_array_B = target_array[:, 2].flatten()
    
    target_array = np.concatenate((target_array_R, target_array_G, target_array_B))

    return input_array, known_array, target_array