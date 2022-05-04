import os
import numpy as np
from PIL import Image


def read_image_using_PIL(img_path):
    # USAGE
    #   np_img, pil_img = read_image_using_PIL(img_path)

    try:
        img = Image.open(img_path)

        np_img = np.asarray(img, dtype=np.uint8)
        pil_img = Image.fromarray(np_img)
    except:
        np_img = None
        pil_img = None

    return np_img, pil_img


def save_pil_img(pil_img, output_dir, new_name):
    """

    """

    new_fpath = os.path.join(output_dir, new_name)

    pil_img.save(new_fpath)

    return new_fpath


def write_to_file(fname, str):
    with open(fname, "a+") as file_object:
        file_object.write(str + "\n")


def validate_images(input_dir, output_dir, log_file, formatter="d"):
    """
    output_dir: image files will be copied here.
    log_file: path of the log file.
    """

    list_files = os.listdir(input_dir)
    list_files = sorted(list_files)

    # if len(list_files) == 0:
    #     print('There is no file in the input directory.')
    #     return

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isfile(log_file):
        log_file = os.path.join(input_dir.split('/')[-1] + '.log') # 'log.txt'
        # input_basename = os.path.basename(input_dir)
        # tmp = os.path.join( input_basename)
        # log_file = tmp + ".log"
        open(log_file, mode='a+').close() # creates an empty file

    num_valid = 0
    copied_files = []

    # for fname in list_files: # E.g., file = '1.jpg' ==> string class
    for dirpath, _, fnames in os.walk(input_dir):
        for fname in sorted(fnames):

            img_path = os.path.join(input_dir, fname)
            abs_file_path = os.path.abspath(img_path)

            if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg'):
                if os.path.getsize(img_path) <= 250000:

                    np_img, pil_img = read_image_using_PIL(img_path)

                    if pil_img is None:
                        # the file could not be read using PIL
                        write_to_file(log_file, abs_file_path + ",3")
                    else:
                        # PIL successfully read the image.

                        if not pil_img.mode.lower() == "rgb" or \
                                pil_img.size[0]<96 or pil_img.size[1]<96:

                            write_to_file(log_file, abs_file_path + ",4")

                        else:

                            if np_img[:,:,0].var() == 0 and np_img[:,:,1].var() == 0 and np_img[:,:,2].var() == 0:  #np_img[0].var() == 0
                                write_to_file(log_file, abs_file_path + ",5")
                            else:

                                if pil_img in copied_files:
                                    write_to_file(log_file, abs_file_path + ",6")
                                else:
                                    try:
                                        new_name = '{:{}}.jpg'.format(num_valid, formatter)
                                    except:
                                        print('ERROR: formatter is incorrect. Use a valid Python formatter. E.g., 06d')
                                        return

                                    new_fpath = os.path.join(output_dir, new_name)
                                    pil_img.save(new_fpath)

                                    copied_files.append(pil_img)

                                    num_valid = num_valid + 1

                else:
                    write_to_file(log_file, abs_file_path + ",2")
            else:
                write_to_file(log_file, abs_file_path + ",1") # absolute file path + comma + error code

    return num_valid

# input_dir = './data'
# output_dir = './data2'
# log_file = ""
# # formatter="06d" # 06d
# formatter = ""
# num_valid = validate_images(input_dir, output_dir, log_file, formatter)
# print("Number of valid image files that were copied to the output directory = ", num_valid)