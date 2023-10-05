"""

An HDF5 file is a container for two kinds of objects:
    datasets  => work like Numpy arrays
    groups    => work like dictionaries

SEE
    https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
"""

import h5py
import numpy as np

f = h5py.File("guide_hdf5_example1.hdf5", "w")

arr1 = np.random.randn(100)
dset = f.create_dataset("my array 1", data=arr1, dtype=arr1.dtype, shape=arr1.shape)
arr2 = np.arange(200)
dset = f.create_dataset("my array 2", data=arr2, dtype=arr2.dtype, shape=arr2.shape)

print("\nIn HDF5 system, datasets work like Numpy arrays:")
print(f"dset={dset}")
print(f"dset.name={dset.name}, dset.shape={dset.shape}, dset.dtype={dset.dtype}")

print("\nIn HDF5 system, groups (think of them like directories) work like dictionaries:")
# The “folders” or "directories" in HDF5 system are
# called groups. The File object itself is a group,
# and in this case, it is a root group, named '/'

print(f"f.name={f.name}")

f.close()

"""
Reading an HDF5 file
"""
with h5py.File("guide_hdf5_example1.hdf5", "r") as f:
    print("Keys are as follows:")
    for key in f.keys():
        print(key)

    print("We can use keys to retrieve data (datasets) stored in an HDF5 file:")
    for key in f.keys():  # EACH KEY IS A DATASET
        data = f[key]
        print(f"data.name={data.name}, data.shape={data.shape}")
        # Note that we are using data as a regular numpy array. Later, we will see
        # that data is pointing to the HDF5 file but is not loaded to memory as a numpy array would.
        # Therefore, DATA here is not a Numpy array. It is a DATASET.

"""
We are not reading the data from the file. Instead, we are generating 
a pointer to where the data is located on the hard drive.
"""
try:
    with h5py.File("guide_hdf5_example1.hdf5", "r") as f:
        data = f["my array 1"]
        print(data[10])
    print(data[10])  # ValueError: Dset_id is not a dataset id (dset_id is not a dataset ID)
except ValueError:
    print("ERROR: This piece of code generates a VALUE ERROR")

with h5py.File("guide_hdf5_example1.hdf5", "r") as f:
    data = f["my array 1"][()]   # Prevents the above ERROR
print(data[10])


"""
Organizing some data in an HDF5 file
"""

my_data = {"A": np.arange(5),
           "B": np.random.randn(5),
           "C": [np.arange(3), np.random.randn(3)]}  # it will concatenate them and forms an array with shape [2, 3]

fname = "guide_hdf5_example2.hdf5"
with h5py.File(fname, "w") as f:

    g = f.create_group("my_data")

    for k in my_data.keys():
        g.create_dataset(k, data=my_data[k])

# Let's add a new dataset to the current file
new_key = "D"
my_data.update({new_key: np.random.randn(100)})

with h5py.File(fname, "a") as f:
    f.create_dataset(new_key, data=my_data[new_key])

# Appending a new dataset to a GROUP in the current file
new_key = "E"
my_data.update({new_key: np.random.randn(200)})

with h5py.File(fname, "a") as f:
    g = f["/my_data"]
    g.create_dataset(new_key, data=my_data[new_key])

# adding a new bunch of keys and elements.
new_my_data = {"F": np.arange(5),
               "G": np.random.randn(5)}

with h5py.File(fname, "a") as f:
    g = f["/my_data"]
    for k in new_my_data:
        if k not in g.keys():
            g.create_dataset(k, data=new_my_data[k])

# Reading all datasets and groups
with h5py.File(fname, "r") as f:
    print("Keys (or datasets) are as follows:")
    for key in f.keys():
        data = f[key]
        if isinstance(f[key], h5py.Dataset):
            print(f"data.name={data.name}, data.shape={data.shape}")
        else:
            g = f[key]
            for gkey in g.keys():
                data = g[gkey]
                print(f"data.name={data.name}, data.shape={data.shape}")

# """
# Append data in numpy arrays
# """
# print("Appending data")
# print(dset.shape)
#
# arr2 = np.ones(10)*2000
# dset = np.append(dset, arr2)
#
# print(dset.shape)
#
# print("test")