import shutil
import os
import numpy as np


def print_exif(img):
    # Found at
    # https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image

    img_exif = img.getexif()

    if img_exif is None:
        print('Sorry, image has no exif data.')
    else:
        for key, val in img_exif.items():
            print(key, val)
                
def copy_files(input_dir, output_dir):
    """
    This function can copy files from input_dir to output_dir

    output_dir must be exist.
    """
    list_files = os.listdir(input_dir)
    list_files = sorted(list_files, reverse = True)

    # k = 0
    for file in list_files: # E.g., file = '1.jpg' ==> string class
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            # check size (bytes) <= 2500000 --- if
                #     rgb?
                      #
            outpath = shutil.copy(input_dir + file, output_dir)
            # file_name = "%d.jpg"%k
            # save your file with "file_name"
            # outpath = shutil.copy(input_dir + file, output_dir + file_name)
            # k = k + 1
            # GIGO = Garbage In ==> Garbage Out
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

print("img.size: ", img.size)
print("img.mode: ", img.mode) # L: gray-scale (Luminance) --- RGB: color image

# Returns the contents of this image as a sequence object containing pixel values.
PIL_sequence_object = img.getdata()
print(type(PIL_sequence_object)) # <class 'ImagingCore'> => this is an internal PIL sequence object.
print(PIL_sequence_object)       # <ImagingCore object at 0x000002153C977C70>
print(PIL_sequence_object[0])    # a tuple is returned. For each pixel, three integer values are stored.


"""
Remove metadata or EXIF
"""
#   1. create a new PIL image - initialized with 0
#   1. convert PIL sequence object into a list
#   2. use PIL's putdata
#   3. save the new image
img2 = Image.new(img.mode, img.size) # Creates a new image with the given mode and size.
print('img2[0]', img2.getdata()[0])
data = list(img.getdata())
img2.putdata(data)
#     This method copies data from a sequence object into
#     the image, starting at the upper left corner (0, 0),
#     and continuing until either the image or the sequence ends.
img2.save(r'./data/1400_noEXIF.jpg')

# print_exif(img)
# print_exif(img2)
print(img.info.keys())


"""
file size in bytes
"""
print(os.path.getsize(r'./data/1400_noEXIF.jpg'))


"""
Remove metadata or EXIF - 2nd way
"""

def copy_pixel_only(img_path, new_name):
    img = Image.open(img_path)
    img_np = np.uint8(np.asarray(img))
    img2 = Image.fromarray(img_np).convert('RGB')

    fpath, fname = os.path.split(img_path)
    print(fpath)
    print(fname)
    new_fpath = os.path.join(fpath, new_name)
    img2.save(new_fpath)
    return new_fpath

img_path = r'./data/image_baby.jpg'
new_name = 'image_baby_noExif.jpg'
copy_pixel_only(img_path, new_name)

# img = Image.open(r'./data/image_baby.jpg') # PIL object ==>
# print(type(img)) # <class 'PIL.JpegImagePlugin.JpegImageFile'> ==> data + functions
#
# img2_np = np.uint8(np.asarray(img)) # convert a PIL object into numpy array
# print("=====> Variance of the image: ", img2_np.var())
# img2 = Image.fromarray(img2_np).convert('RGB')
# img2.save(r'./data/image_baby_noExif.jpg')



"""
format string / numbers
"""
# In your assignment, you have a list of files.
# E.g., ['dog1.jpg', 'cat.jpg'] === your copy function ===> 0.jpg, 1.jpg (F
formatter = ""     # 0, 1, 2, 3, 4, ...
formatter = "30d"  # 000, 001, ....

for i in range(5):
    print("%d"%i)
    print("%3.d" % i)
    print("%3d" % i)
    print("%.3d" % i)
    print("%.6d.jpg"%i)
    print("")


"""
Given pixel values ===> create an image 
"""
pixel_values = np.ones(shape = (512,512), dtype=np.int32)*150
img_np = np.uint8(np.asarray(pixel_values))
img_pil = Image.fromarray(img_np).convert('L')
img_pil.save(r'./data/constant_image.jpg')

print("=====> Variance of the image: ", img_np.var())


pixel_values2 = np.ones(shape = (512,512,3), dtype=np.int32)*150
pixel_values2[:,:,1] = np.ones(shape = (512,512), dtype=np.int32)*10
pixel_values2[:,:,2] = np.ones(shape = (512,512), dtype=np.int32)*80
img_np = np.uint8(np.asarray(pixel_values2))
img_pil = Image.fromarray(img_np).convert('RGB')
img_pil.save(r'./data/constant_image_rgb.jpg')


# 
#  if os.path.getsize(r'./data/1400_noEXIF.jpg')<= 2500000:
#  ......

def write_to_file(fname, str):
    with open(fname, "a+") as file_object:
        file_object.write(str + "\n")

write_to_file('./data/test.txt', 'Hello world!')
write_to_file('./data/test.txt', 'Hello Soheil.')