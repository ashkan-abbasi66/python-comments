"""
To avoid reading each Numpy file (or image) directly from an individual file, this script
saves all (current OCT, current VF) pairs inside ONE BIG HDF5 FILE, which stores in
FINAL_DATASET_PATH.

To use this dataset, you need to define a special dataset object.

DESCRIPTION:
    In HDF5 format, every thing can be stored inside "groups" and "datasets".
    Here, the name of each CSV file is used to create a groups. Then, sample_id
    of each record is used to create the required "datasets" in the HDF5 format.
        1000_image
        1000_vf
Observation:
I found that reading individual Numpy files from an SSD drive using the iterator
defined in this file is faster than reading the contents of those files from one 
big HDF5 file (with or without compression). Note that essentially this script
stores the content of each Numpy files inside one HDF5 file and uses the file names
as groups to access their contents.
"""
import time

import h5py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


NUM_FILES_TO_STORE_AND_READ = 20

class GetImageVF:
    def __init__(self, df, oct_image_dir):

        self.sample_id_col = df.columns.get_loc("sample_id")
        self.eye_col = df.columns.get_loc("eye")
        self.filename_col = df.columns.get_loc("corresponding_OCT_filename")
        self.vf_start_col = df.columns.get_loc("current_vf_1")
        self.vf_end_col = df.columns.get_loc("current_vf_54") + 1

        self.where_files_stored = oct_image_dir

        self.df = df

        self.counter = 0

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        return self

    def __next__(self):

        item = self.counter

        image_filename = self.df.iloc[item, self.filename_col]
        sample_id = int(self.df.iloc[item, self.sample_id_col])

        image_filepath = os.path.join(self.where_files_stored, image_filename)

        try:
            x = np.load(image_filepath)
        except:
            raise Exception("There is a problem in loading this file:\n\t %s" % image_filename)

        # DEBUG:
        # import matplotlib.pyplot as plt; plt.figure(); plt.imshow(x[35, :, :], cmap="gray"); plt.show()

        # target values as 1D vector
        y = self.df.iloc[item, self.vf_start_col: self.vf_end_col].to_numpy().astype(np.float32)
        y = y / 48.

        self.counter += 1

        return sample_id, x, y


def store_image_vf_pairs_inside_hdf5_file(csvfile_path, ff, DEBUG=False):
    df = pd.read_csv(csvfile_path)
    gg = ff.create_group(csv_filename)

    getimage_obj = GetImageVF(df, oct_image_dir)
    get_input_target = iter(getimage_obj)

    if DEBUG:
        # Only extracts a limited number of samples from each iterator (or CSV file)
        counter = 1

    for _ in tqdm(range(len(get_input_target))):
        sample_id, x, y = next(get_input_target)
        gg.create_dataset(f"{sample_id}_image", data=x, dtype=x.dtype,
                          # compression="gzip"
                          )
        gg.create_dataset(f"{sample_id}_vf", data=y, dtype=y.dtype,
                          # compression="gzip"
                          )

        if DEBUG:
            counter += 1
            if counter >= NUM_FILES_TO_STORE_AND_READ:
                break


if __name__ == '__main__':

    DEBUG = True

    data_dir_name = "vf-estimation-data"

    # Input-target pairs are determined by CSV files stored here:
    data_dir = f"../{data_dir_name}/NYUPITT_10fold_OCTFilename_VF"

    # Individual down-sampled OCT files were stored as Numpy files here:
    oct_image_dir = "../vf-forecast-cnn-data/NYUPITT_72x144x72_NPY"

    # The output of this script will be a dataset stored here:
    to_save_dir = f"../{data_dir_name}/"

    if DEBUG:
        oct_image_dir = "e:/NYUPITT_72x144x72_NPY"
        to_save_dir = f"e:/{data_dir_name}/"

    if not os.path.exists(to_save_dir):
        os.makedirs(to_save_dir)

    k_fold = 10
    FINAL_DATASET = "NYUPITT_72x144x72_VF_pairs.hdf5"
    FINAL_DATASET_PATH = os.path.join(to_save_dir, FINAL_DATASET)

    with h5py.File(FINAL_DATASET_PATH, "w") as ff:
        for fold_number in range(1, k_fold+1):
            csv_filename = "fold%d_train.csv" % fold_number
            store_image_vf_pairs_inside_hdf5_file(os.path.join(data_dir, csv_filename), ff, DEBUG)

            csv_filename = "fold%d_val.csv" % fold_number
            store_image_vf_pairs_inside_hdf5_file(os.path.join(data_dir, csv_filename), ff, DEBUG)

            csv_filename = "fold%d_test.csv" % fold_number
            store_image_vf_pairs_inside_hdf5_file(os.path.join(data_dir, csv_filename), ff, DEBUG)

    if DEBUG:
        with h5py.File(FINAL_DATASET_PATH, "r") as ff:
            print("Keys (or datasets) are as follows:")
            for key in ff.keys():
                if isinstance(ff[key], h5py.Dataset):
                    print(f"ff[key].name={ff[key].name}, data.shape={ff[key].shape}")
                else:
                    gg = ff[key]
                    for gkey in gg.keys():
                        print(f"gg[gkey].name={gg[gkey].name}, gg[gkey].shape={gg[gkey].shape}")

    if DEBUG:
        # speed test
        start_time = time.time()
        with h5py.File(FINAL_DATASET_PATH, "r") as ff:
            for key in ff.keys():
                gg = ff[key]
                # print("Processing ", gg, "...")
                for gkey in gg.keys():
                    data = np.array(list(gg[gkey]))
                    data += 10
        end_time = time.time() - start_time
        print("Elapsed time:", end_time)  # Elapsed time: 1.882068157196045


        def read_samples(df, oct_image_dir, DEBUG):
            getimage_obj = GetImageVF(df, oct_image_dir)
            get_input_target = iter(getimage_obj)

            if DEBUG:
                # Only extracts a limited number of samples from each iterator (or CSV file)
                counter = 1

            for _ in range(len(get_input_target)):
                sample_id, x, y = next(get_input_target)
                x += 10

                if DEBUG:
                    counter += 1
                    if counter >= NUM_FILES_TO_STORE_AND_READ:
                        break


        total_time = 0
        for fold_number in range(1, k_fold+1):
            start_time = time.time()
            csv_filename = "fold%d_train.csv" % fold_number
            df = pd.read_csv(os.path.join(data_dir, csv_filename))
            # start_time = time.time()
            read_samples(df, oct_image_dir, DEBUG)
            end_time = time.time() - start_time

            total_time += end_time

            start_time = time.time()
            csv_filename = "fold%d_val.csv" % fold_number
            df = pd.read_csv(os.path.join(data_dir, csv_filename))
            # start_time = time.time()
            read_samples(df, oct_image_dir, DEBUG)
            end_time = time.time() - start_time

            total_time += end_time

            start_time = time.time()
            csv_filename = "fold%d_test.csv" % fold_number
            df = pd.read_csv(os.path.join(data_dir, csv_filename))
            # start_time = time.time()
            read_samples(df, oct_image_dir, DEBUG)
            end_time = time.time() - start_time

            total_time += end_time

        print("Elapsed time:", total_time)  #
