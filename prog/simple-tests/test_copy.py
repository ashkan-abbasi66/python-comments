import shutil
import os

def copy_files(input_dir, output_dir):
    """
    This function can copy files from input_dir to output_dir

    output_dir must be exist.
    """
    list_files = os.listdir(input_dir)
    list_files = sorted(list_files, reverse = True)

    for file in list_files: # E.g., file = '1.jpg' ==> string class
        if file.endswith('.jpg') or file.endswith('.tif'):
            outpath = shutil.copy(input_dir + file, output_dir)
            print(outpath)
        # if file.endswith('.tif'):
        #     outpath = shutil.copy(input_dir + file, output_dir)
        #     print(outpath)
        # else:
        #     print(input_dir + file, " is not a .jpg file.")

# def copy_raw_data(input_dir, output_dir):
#     """
#     Images ==> PIL read ==>
#
#     """
#     pass

# MAIN PROGRAM
# input_dir = './data/'
# output_dir = './data2'
#
# if not os.path.isdir(output_dir):
#     os.makedirs(output_dir)
#
# copy_files(input_dir, output_dir)

from PIL import Image

img = Image.open(r'./data/1400.jpg')

# Get raw
data = list(img.getdata()) # img.getdata() => Returns the contents of this image as a sequence object