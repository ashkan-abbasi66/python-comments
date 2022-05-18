import numpy as np
from PIL import Image

def ex4(image_array, offset, spacing):
    """
    image_array: RGB image - numpy array (M, N, 3)
    offset: a tuple with 2 integer values
    spacing: a tuple with 2 integer values
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("\"image_array\" should contain a numpy array")

    if not (len(image_array.shape) == 3 and image_array.shape[2] == 3):
        raise NotImplementedError("\"image_array\" should contain an image with 3 channels.")

    try:
        int(offset[0])
        int(offset[1])
        int(spacing[0])
        int(spacing[1])
    except ValueError:
        raise ValueError("\"offset\" or \"spacing\" value(s) are not convertible to int objects")

    if offset[0] < 0 or offset[0] > 32:
        raise ValueError("\"offset\" should be in range(0, 33)")
    if spacing[0] < 2 or spacing[0] > 8:
        raise ValueError("\"spacing\" should be in range(2, 9)")

    known_array = np.zeros_like(image_array)

    M,N,C = image_array.shape

    for j in range(offset[0], N, spacing[0]):
        for i in range(offset[1], M, spacing[1]):
            known_array[i, j, :] = 1
    print("INFO: number of remaining pixels: ", known_array[:,:,0].sum())

    # comment out the following, if you want to perform a check on small images
    if known_array[:,:,0].sum() < 144:
        raise ValueError("Number of the remaining pixels is smaller than 144.")

    input_array = np.multiply(image_array, known_array)
    target_array = image_array.copy()

    # H*W*C ===> C*H*W
    axes = (2, 0, 1)
    input_array = np.transpose(input_array, axes)
    known_array = np.transpose(known_array, axes)

    R_channel = target_array[:, :, 0].flatten()
    G_channel = target_array[:, :, 1].flatten()
    B_channel = target_array[:, :, 2].flatten()

    target_array = np.concatenate([R_channel, G_channel, B_channel])

    return input_array, known_array, target_array


if __name__ == "__main__":

    # ############ Note  #############
    # To run these tests comment out checking parts of the ex4 function

    # ############ test 1 ############
    # test_file = './data/mountain.jpg'
    # im_pil = Image.open(test_file)
    # im = np.asarray(im_pil)
    # im = im[0:10,0:10,:]

    # ############ test 2 ############
    test_shape = (10, 10, 3)
    im_rgb = np.empty(test_shape)
    im_rgb[:, :, 0] = np.ones(test_shape[:2])
    im_rgb[:, :, 1] = np.ones(test_shape[:2])*2.0
    im_rgb[:, :, 2] = np.ones(test_shape[:2])*3.0
    im = im_rgb.astype(np.float)
    # ###############################

    print("input data shape:", im.shape)

    offset = (2,1)
    spacing = (2,3)
    input_array, known_array, target_array = ex4(im, offset, spacing)

    print(input_array.shape)
    print(known_array.shape)

    print("r\n", input_array[0,:,:])
    print("g\n", input_array[1, :, :])
    print("b\n", input_array[2, :, :])

    print("target_array size:\n", target_array.shape)

    # data should be printed in the order of R, G, and B
    print(input_array[known_array < 1])