"""

"""
import os
from PIL import Image
import numpy as np


class ImageStandardizer:
    def __init__(self, input_dir):
        """
        Algorithm:

        1. Scan directory recursively
        2. Raise a ValueError if there are no .jpg files
        3. Transform all paths to absolute paths and sort them alphabetically in ascending order.
        4. Store the sorted absolute file paths in an attribute self.files

        Arguments:
        input_dir: relative or abs path to the input directory
        """

        self.files = []
        self.mean = None
        self.std = None
        self.input_dir_abs = os.path.abspath(input_dir)

        jpg_counter = 0
        for root, dir_names, file_names in os.walk(self.input_dir_abs):
            for file_name in file_names:
                if file_name.endswith('.jpg'):
                    self.files.append(os.path.join(root, file_name))
                    jpg_counter += 1

        if jpg_counter == 0:
            msg = "There are no .JPG files."
            raise ValueError(msg)

        self.files = sorted(self.files)

        print('==> List of *.JPG files:')
        for f in self.files:
            print(f)

    def analyze_images(self):
        """
        Compute the means and standard deviations for each color channel of all images
        self.mean = The average over the RGB means of all images
        self.std = The average over the RGB STDs of all images

        Return the tuple (self.mean, self.std)
        """
        try:

            rgb_counter = 0
            to_pop = []

            self.mean = np.array([0, 0, 0], dtype=np.float64)  # #########################
            self.std = np.array([0, 0, 0], dtype=np.float64)

            for pos, file in enumerate(self.files):

                img = Image.open(file)

                if img.mode.lower() == 'rgb':

                    np_img = np.asarray(img, dtype=np.float64)

                    rgb_means = [np_img[:, :, i].mean() for i in range(3)]
                    rgb_std = [np_img[:, :, i].std() for i in range(3)]

                    # print('RGB means are: (file name: %s)' % os.path.basename(file))
                    # print(rgb_means)

                    rgb_counter += 1
                    self.mean = self.mean + np.array(rgb_means)
                    self.std = self.std + np.array(rgb_std)

                else:
                    to_pop.append(pos)

            if rgb_counter == 0:
                raise ValueError("There are no RGB files.")

            for p in to_pop:
                self.files.pop(p)

            print("==> Number of processed images: ", rgb_counter)

            self.mean = self.mean / rgb_counter
            self.std = self.std / rgb_counter

        except:
            raise ValueError('PIL cannot read this file: %s' % file)

        return self.mean, self.std # #########################

    def get_standardized_images(self):
        """
        return numpy array of the images in the self.files

        """

        if self.mean is None or self.std is None:
            raise ValueError("self.mean or self.std is None!")

        for file in self.files:
            img = Image.open(file)
            np_img = np.asarray(img, dtype=np.float32) # #########################
            assert np_img.shape[2] == 3, "The image is not RGB"

            np_img = (np_img - self.mean) / self.std

            yield np_img.astype(np.float32)



# data_dir = './data'
# jpg_images = ImageStandardizer(data_dir)
# jpg_images.analyze_images()
#
# print("Mean vector of all images:")
# print(jpg_images.mean)
# print(jpg_images.std, '\n')
#
# im_gen = jpg_images.get_standardized_images()
# # print(next(im_gen).shape)
# for im in im_gen:
#     print(im.shape)
