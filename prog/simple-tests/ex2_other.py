def validate_images(input_dir, output_dir, log_file, formatter=None):
    import os
    import shutil
    from PIL import Image, UnidentifiedImageError, ImageStat
    import numpy as np
    
    input_dir = input_dir.replace('\\', '/')
    output_dir = output_dir.replace('\\', '/')
    
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # Get filepaths
    basenames = []
    abspaths = []
    for dirpath, _, filenames in os.walk(input_dir):
        for f in sorted(filenames):
            basenames.append(f)
            absp = os.path.abspath(os.path.join(dirpath, f))
            absp = absp.replace('\\', '/')
            abspaths.append(absp)

    basenames = np.array(basenames)
    abspaths = np.array(abspaths)
    sorted_index = np.arange(len(basenames))

    ### Validity check
    valid_file_indices = []

    invalid_files = []
    log_file_lines = []
    invalid_file_indices = []
    
    checked_open_images = []
    for i in sorted_index:
        file_name = basenames[i]

        # Name ends
        valid_or_not = []
        if file_name.endswith('.jpg') or file_name.endswith('.JPG') \
            or file_name.endswith('.jpeg') or file_name.endswith('.JPEG'):
            pass
        else:
            valid_or_not.append(1)

        # File size
        if os.path.getsize(abspaths[i]) > 250000:
            valid_or_not.append(2)

        # Can be read as image
        try:
            Image.open(abspaths[i])
        except UnidentifiedImageError:
            valid_or_not.append(3)
        else:
            
            # Image shape
            opened_image = Image.open(abspaths[i])
            numpy_image_array = np.asarray(opened_image)
            if numpy_image_array.ndim != 3:
                valid_or_not.append(4)
            else:
                H, W, D = numpy_image_array.shape
                if H >= 96 and W >= 96 and D == 3:
                    pass
                else:
                    valid_or_not.append(4)

                # Variance
                image_array = np.array(opened_image)
                image_array = image_array.reshape(-1, 3)
                calc_variance = np.var(image_array, axis=0)
                ideal_variance = np.array([0, 0, 0])
                close = np.allclose(calc_variance, ideal_variance, atol=1e-7)
                if close:
                    valid_or_not.append(5)

                # File not copied
                checker = False
                list_img_now = list(opened_image.getdata())
                for list_img in checked_open_images:
                    if list_img_now == list_img:
                        checker = True
                if checker:
                    valid_or_not.append(6)
                checked_open_images.append(list_img_now)

        # Store invalid filenames
        if len(valid_or_not) != 0:
            invalid_files.append(abspaths[i])
            log_file_lines.append(abspaths[i].split(input_dir)[1][1:]+';'+str(min(valid_or_not)))
            invalid_file_indices.append(i)
        else:
            valid_file_indices.append(i)

    # Apply formatter
    formatted = []
    if formatter != None:
        length = int(formatter[:-1])
        for i in range(len(valid_file_indices)):
            n = len(str(i))
            new_name = '0'*(length - n) + str(i) + '.jpg'
            formatted.append(new_name)
    else:
        for i in range(len(valid_file_indices)):
            new_name = str(i) + '.jpg'
            formatted.append(new_name)
            
    # Copy files
    for n in range(len(valid_file_indices)):
        i = valid_file_indices[n]
        shutil.copyfile(abspaths[i], output_dir+'/'+formatted[n])
        
    with open(log_file, 'w') as l:        
        content = '\n'.join(log_file_lines)+'\n'
        if len(log_file_lines) > 0:
            l.write(content)

    return len(valid_file_indices)

####### TEST

# input_dir = "unittest_\\unittest_input_8"
# output_dir = "unittest_\\outputs\\unittest_input_8"
# log_file = "unittest_\\outputs\\unittest_input_8.log"
# formatter = "06d"

# valids = validate_images(input_dir, output_dir, log_file, formatter=formatter)
# print(valids)


input_dir = "data\\"
output_dir = "data2\\"
log_file = ".\\data2\\log.txt"
formatter = "06d"

valids = validate_images(input_dir, output_dir, log_file, formatter=formatter)
print(valids)