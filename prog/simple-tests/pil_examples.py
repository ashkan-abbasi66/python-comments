from PIL import Image
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
# See also "pil_copy_valid_images.py" for a simpler way
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